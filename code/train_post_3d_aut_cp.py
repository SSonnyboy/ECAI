import argparse
import logging
import os
import os.path as osp
import random
import sys
import yaml

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from bcp import *
from dataloaders.mixaugs import cut_mix_3d
from dataloaders.dataset_3d import (
    LAHeart,
    Pancreas,
    WeakStrongAugment,
    TwoStreamBatchSampler,
)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.util import update_values, time_str, AverageMeter
from val_3D import var_all_case_LA, var_all_case_Pancrease
from at import *
from EBS import *


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        I. helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


def get_rampup_param(iter, max_iter):
    mu_linear = 2 * iter / max_iter - 1
    b_linear = 1 - iter / max_iter
    return mu_linear, b_linear


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        II. trainer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def train(args, snapshot_path):
    model_t1, model_t2 = args["model1"], args["model2"]
    base_lr = args["base_lr"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    num_classes = args["num_classes"]
    at_weight = args["at_weight"]
    alpha = args["alpha"]
    cur_time = time_str()
    writer = SummaryWriter(snapshot_path + "/log")
    csv_train = os.path.join(
        snapshot_path, "log", "seg_{}_train_iter.csv".format(cur_time)
    )
    csv_test = os.path.join(
        snapshot_path, "log", "seg_{}_validate_ep.csv".format(cur_time)
    )

    def worker_init_fn(worker_id):
        random.seed(args["seed"] + worker_id)

    # + + + + + + + + + + + #
    # 1. create model
    # + + + + + + + + + + + #
    model1 = net_factory(net_type=model_t1, in_chns=1, class_num=num_classes)
    model2 = net_factory(net_type=model_t2, in_chns=1, class_num=num_classes)
    model1.cuda()
    model2.cuda()
    model1.train()
    model2.train()

    # + + + + + + + + + + + #
    # 2. dataset
    # + + + + + + + + + + + #
    fdloader = LAHeart
    flag_pancreas = True if "pancreas" in args["root_path"].lower() else False
    if flag_pancreas:
        fdloader = Pancreas
    db_train = fdloader(
        base_dir=args["root_path"],
        split="train",
        num=None,
        transform=transforms.Compose(
            [WeakStrongAugment(args["patch_size"], flag_rot=not flag_pancreas)]
        ),
    )

    labeled_idxs = list(range(0, args["labeled_num"]))
    unlabeled_idxs = list(range(args["labeled_num"], args["max_samples"]))

    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"]
    )

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=args["workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    logging.info("{} iterations per epoch".format(len(trainloader)))

    # + + + + + + + + + + + #
    # 4. optim, scheduler
    # + + + + + + + + + + + #
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=base_lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=base_lr)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # + + + + + + + + + + + #
    # 5. training loop
    # + + + + + + + + + + + #
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance_1 = 0.0
    best_performance_2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    control_gate = 0
    for epoch_num in iterator:
        # metric indicators
        meter_sup_losses1 = AverageMeter()
        meter_uns_losses1 = AverageMeter(20)
        meter_train_losses1 = AverageMeter(20)
        meter_sup_losses2 = AverageMeter()
        meter_uns_losses2 = AverageMeter(20)
        meter_train_losses2 = AverageMeter(20)
        meter_learning_rates = AverageMeter()

        for i_batch, sampled_batch in enumerate(trainloader):
            num_lb = args["labeled_bs"]
            num_ulb = batch_size - num_lb

            # 1) get augmented data
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            img_lb_w, target_lb = weak_batch[num_ulb:], label_batch[num_ulb:]
            img_ulb_w, img_ulb_s = weak_batch[:num_ulb], strong_batch[:num_ulb]

            # 4) forward
            img = torch.cat((img_lb_w, img_ulb_w))
            pred1, g1_1, g2_1, g3_1, _ = model1(img, is_feature=True)
            pred_lb1 = pred1[: args["labeled_bs"]]
            pred_ulb1 = pred1[args["labeled_bs"] :]

            img = torch.cat((img_lb_w, img_ulb_s))
            pred2, g1_2, g2_2, g3_2, _ = model2(img, is_feature=True)
            pred_lb2 = pred2[: args["labeled_bs"]]
            pred_ulb2 = pred2[args["labeled_bs"] :]

            pred_ulb1_soft = torch.softmax(pred_ulb1, dim=1)
            pred_ulb2_soft = torch.softmax(pred_ulb2, dim=1)

            # 5) supervised loss
            loss_lb1 = (
                ce_loss(pred_lb1, target_lb.long())
                + dice_loss(
                    torch.softmax(pred_lb1, dim=1),
                    target_lb.unsqueeze(1).float(),
                )
            ) / 2.0

            loss_lb2 = (
                ce_loss(pred_lb2, target_lb.long())
                + dice_loss(
                    torch.softmax(pred_lb2, dim=1),
                    target_lb.unsqueeze(1).float(),
                )
            ) / 2.0

            # 6) unsupervised loss
            if control_gate == 1:  # performance1>performance2
                at_loss_g1 = at_loss(g1_1.detach(), g1_2)
                at_loss_g2 = at_loss(g2_1.detach(), g2_2)
                at_loss_g3 = at_loss(g3_1.detach(), g3_2)
                at_loss_m2 = 0.1 * at_loss_g1 + 0.3 * at_loss_g2 + 0.6 * at_loss_g3
            elif control_gate == -1:
                at_loss_g1 = at_loss(g1_1, g1_2.detach())
                at_loss_g2 = at_loss(g2_1, g2_2.detach())
                at_loss_g3 = at_loss(g3_1, g3_2.detach())
                at_loss_m1 = 0.1 * at_loss_g1 + 0.3 * at_loss_g2 + 0.6 * at_loss_g3

            # unsup loss
            pseudo_output_soft = (pred_ulb1_soft + pred_ulb2_soft) / 2  # # label smooth
            entropy = -torch.sum(
                pseudo_output_soft * torch.log(pseudo_output_soft + 1e-10), dim=1
            )  # cal entropy
            N = num_classes
            max_entropy = torch.log(torch.tensor(N, dtype=torch.float32))
            normalized_entropy = entropy / max_entropy
            mu_param, b_param = get_rampup_param(iter_num, max_iterations)
            weight_map = mu_param * normalized_entropy.unsqueeze(1) + b_param
            if not flag_pancreas:
                pseudo_output_post = get_LA_masks(pseudo_output_soft)
            else:
                pseudo_output_post = get_pan_mask(pseudo_output_soft)

            loss_ulb1 = dice_loss(
                pred_ulb1_soft,
                pseudo_output_post.unsqueeze(1).float().detach(),
                weight_map=weight_map.detach(),
            )
            loss_ulb2 = dice_loss(
                pred_ulb2_soft,
                pseudo_output_post.unsqueeze(1).float().detach(),
                weight_map=weight_map.detach(),
            )

            # cutmix loss
            if alpha != 1.0:
                img_lb_a, img_lb_b = img_lb_w.chunk(2)
                target_a, target_b = target_lb.chunk(2)
                img_ulb_a, img_ulb_b = img_ulb_w.chunk(2)
                pseudo_a, pseudo_b = pseudo_output_post.detach().chunk(2)
                map_a, map_b = weight_map.detach().chunk(2)
                if flag_pancreas:
                    img_mask, loss_mask = generate_mask_PAN(img_lb_w)
                else:
                    img_mask, loss_mask = generate_mask_LA(img_lb_w)
                net_input_unl = img_ulb_a * img_mask + img_lb_a * (1 - img_mask)
                net_input_l = img_lb_b * img_mask + img_ulb_b * (1 - img_mask)
                out_unl_1, out_l_1 = model1(
                    torch.cat((net_input_unl, net_input_l))
                ).chunk(2)
                cm_loss1_un = mix_loss_3d(
                    out_unl_1,
                    pseudo_a,
                    target_a,
                    loss_mask,
                    weight_map_ul=map_a,
                    unlab=True,
                )
                cm_loss1_l = mix_loss_3d(
                    out_l_1,
                    target_b,
                    pseudo_b,
                    loss_mask,
                    weight_map_ul=map_b,
                )
                cm_loss1 = (cm_loss1_un + cm_loss1_l) / 2
                # print(cm_loss1_un)
                writer.add_scalar("info/loss_cm1", cm_loss1, iter_num)
                out_unl_2, out_l_2 = model2(
                    torch.cat((net_input_unl, net_input_l))
                ).chunk(2)
                cm_loss2_un = mix_loss_3d(
                    out_unl_2,
                    pseudo_a,
                    target_a,
                    loss_mask,
                    unlab=True,
                    weight_map_ul=map_a,
                )
                cm_loss2_l = mix_loss_3d(
                    out_l_2,
                    target_b,
                    pseudo_b,
                    loss_mask,
                    weight_map_ul=map_b,
                )
                cm_loss2 = (cm_loss2_un + cm_loss2_l) / 2
                writer.add_scalar("info/loss_cm2", cm_loss2, iter_num)

            # 7) total loss
            consistency_weight = get_current_consistency_weight(iter_num // 150, args)
            if alpha != 1.0:
                loss_ulb2 = alpha * loss_ulb2 + (1 - alpha) * cm_loss2
                loss_ulb1 = alpha * loss_ulb1 + (1 - alpha) * cm_loss1
            if control_gate == 1:
                loss1 = loss_lb1 + consistency_weight * loss_ulb1
                loss2 = (
                    loss_lb2 + consistency_weight * loss_ulb2 + at_weight * at_loss_m2
                )
                writer.add_scalar("info/loss_at", at_loss_m2, iter_num)
                # writer.add_scalar("info/at_weight", at_weight, iter_num)
            elif control_gate == -1:
                loss1 = (
                    loss_lb1 + consistency_weight * loss_ulb1 + at_weight * at_loss_m1
                )
                loss2 = loss_lb2 + consistency_weight * loss_ulb2
                writer.add_scalar("info/loss_at", at_loss_m1, iter_num)
                # writer.add_scalar("info/at_weight", at_weight, iter_num)
            else:
                loss1 = loss_lb1 + consistency_weight * loss_ulb1
                loss2 = loss_lb2 + consistency_weight * loss_ulb2

            # 8) update student model
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # 10) udpate learing rate
            if args["poly"]:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for optimizer in [optimizer1, optimizer2]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_
            else:
                lr_ = base_lr

            # 11) record statistics
            iter_num = iter_num + 1
            # --- a) writer
            # writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/loss1", loss1, iter_num)
            writer.add_scalar("info/loss2", loss2, iter_num)
            writer.add_scalar("info/loss_lb1", loss_lb1, iter_num)
            writer.add_scalar("info/loss_ulb1", loss_ulb1, iter_num)
            writer.add_scalar("info/loss_lb2", loss_lb2, iter_num)
            writer.add_scalar("info/loss_ulb2", loss_ulb2, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            # --- b) loggers
            logging.info(
                "iteration:{}  t-loss1/2:{:.4f}/{:.4f}, loss-lb1/2:{:.4f}/{:.4f}, loss-ulb1/2:{:.4f}/{:.4f}, weight:{:.2f}, lr:{:.4f}".format(
                    iter_num,
                    loss1.item(),
                    loss2.item(),
                    loss_lb1.item(),
                    loss_lb2.item(),
                    loss_ulb1.item(),
                    loss_ulb2.item(),
                    consistency_weight,
                    lr_,
                )
            )
            # --- c) avg meters
            meter_sup_losses1.update(loss_lb1.item())
            meter_uns_losses1.update(loss_ulb1.item())
            meter_sup_losses2.update(loss_lb2.item())
            meter_uns_losses2.update(loss_ulb2.item())
            meter_train_losses1.update(loss1.item())
            meter_train_losses2.update(loss2.item())
            meter_learning_rates.update(lr_)

            # --- d) csv
            tmp_results = {
                "loss1": loss1.item(),
                "loss2": loss2.item(),
                "loss_lb1": loss_lb1.item(),
                "loss_lb2": loss_lb2.item(),
                "loss_ulb1": loss_ulb1.item(),
                "loss_ulb2": loss_ulb2.item(),
                "lweight_ub": consistency_weight,
                "lr": lr_,
            }
            data_frame = pd.DataFrame(
                data=tmp_results, index=range(iter_num, iter_num + 1)
            )
            if iter_num > 1 and osp.exists(csv_train):
                data_frame.to_csv(csv_train, mode="a", header=None, index_label="iter")
            else:
                data_frame.to_csv(csv_train, index_label="iter")

            if iter_num >= max_iterations:
                break

        # 12) validating
        if (
            epoch_num % args.get("test_interval_ep", 1) == 0
            or iter_num >= max_iterations
        ):
            model1.eval()
            model2.eval()

            if "pancreas" in args["root_path"].lower():
                performance_1 = var_all_case_Pancrease(
                    model1,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=16,
                    stride_z=16,
                    flag_nms=True,
                )
                performance_2 = var_all_case_Pancrease(
                    model2,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=16,
                    stride_z=16,
                    flag_nms=True,
                )
            else:
                performance_1 = var_all_case_LA(
                    model1,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=18,
                    stride_z=4,
                )
                performance_2 = var_all_case_LA(
                    model2,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=18,
                    stride_z=4,
                )

            if performance_1 > best_performance_1:
                best_performance_1 = performance_1
                tmp_model1_snapshot_path = os.path.join(snapshot_path, model_t1 + "_1")
                if not os.path.exists(tmp_model1_snapshot_path):
                    os.makedirs(tmp_model1_snapshot_path, exist_ok=True)
                save_mode_path_stu = os.path.join(
                    tmp_model1_snapshot_path,
                    "ep_{:0>3}_dice_{}.pth".format(
                        epoch_num, round(best_performance_1, 4)
                    ),
                )
                torch.save(model1.state_dict(), save_mode_path_stu)

                save_best_path_stu = os.path.join(
                    snapshot_path, "best_{}_model1.pth".format(model_t1)
                )
                torch.save(model1.state_dict(), save_best_path_stu)

            if performance_2 > best_performance_2:
                best_performance_2 = performance_2
                tmp_model2_snapshot_path = os.path.join(snapshot_path, model_t2 + "_2")
                if not os.path.exists(tmp_model2_snapshot_path):
                    os.makedirs(tmp_model2_snapshot_path, exist_ok=True)
                save_mode_path = os.path.join(
                    tmp_model2_snapshot_path,
                    "ep_{:0>3}_dice_{}.pth".format(
                        epoch_num, round(best_performance_2, 4)
                    ),
                )
                torch.save(model2.state_dict(), save_mode_path)

                save_best_path = os.path.join(
                    snapshot_path, "best_{}_model2.pth".format(model_t2)
                )
                torch.save(model2.state_dict(), save_best_path)
            # compare the performance
            if performance_1 > performance_2:
                control_gate = 1
            elif performance_1 < performance_2:
                control_gate = -1
            else:
                control_gate = 0
            # writer
            writer.add_scalar("Var_dice/Dice_1", performance_1, epoch_num)
            writer.add_scalar("Var_dice/Best_dice_1", best_performance_1, epoch_num)
            writer.add_scalar("Var_dice/Dice_2", performance_2, epoch_num)
            writer.add_scalar("Var_dice/Best_dice_2", best_performance_2, epoch_num)

            # csv
            tmp_results_ts = {
                "loss_total1": meter_train_losses1.avg,
                "loss_total2": meter_train_losses2.avg,
                "loss_sup1": meter_sup_losses1.avg,
                "loss_unsup1": meter_uns_losses1.avg,
                "loss_sup2": meter_sup_losses2.avg,
                "loss_unsup2": meter_uns_losses2.avg,
                "learning_rate": meter_learning_rates.avg,
                "Dice_1": performance_1,
                "Dice_1_best": best_performance_1,
                "Dice_2": performance_2,
                "Dice_2_best": best_performance_2,
            }
            data_frame = pd.DataFrame(
                data=tmp_results_ts, index=range(epoch_num, epoch_num + 1)
            )
            if epoch_num > 0 and osp.exists(csv_test):
                data_frame.to_csv(csv_test, mode="a", header=None, index_label="epoch")
            else:
                data_frame.to_csv(csv_test, index_label="epoch")

            # logs
            logging.info(
                " <<Test>> - Ep:{}  - Dice-1/2:{:.2f}/{:.2f}, Best-1:{:.2f}, Best-2:{:.2f}".format(
                    epoch_num,
                    performance_1 * 100,
                    performance_2 * 100,
                    best_performance_1 * 100,
                    best_performance_2 * 100,
                )
            )
            logging.info(
                "          - AvgLoss1(lb/ulb/all):{:.4f}/{:.4f}/{:.4f}- AvgLoss2(lb/ulb/all):{:.4f}/{:.4f}/{:.4f}".format(
                    meter_sup_losses1.avg,
                    meter_uns_losses1.avg,
                    meter_train_losses1.avg,
                    meter_sup_losses2.avg,
                    meter_uns_losses2.avg,
                    meter_train_losses2.avg,
                )
            )

            model1.train()
            model2.train()

        if (epoch_num + 1) % args.get("save_interval_epoch", 1000000) == 0:
            save_mode_path = os.path.join(
                snapshot_path, "epoch_" + str(epoch_num) + ".pth"
            )
            torch.save(model1.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        III. main process
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if __name__ == "__main__":
    # 1. set up config
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="", help="configuration file")

    # Basics: Data, results, model
    parser.add_argument(
        "--root_path", type=str, default="./data/LA", help="Name of Experiment"
    )
    parser.add_argument(
        "--res_path", type=str, default="./results/LA", help="Path to save resutls"
    )
    parser.add_argument("--exp", type=str, default="LA/POST", help="experiment_name")
    parser.add_argument("--model1", type=str, default="res18vnet")
    parser.add_argument("--model2", type=str, default="res34vnet")
    parser.add_argument(
        "--at_weight", type=float, default=10, help="output channel of network"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="output channel of network"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="the id of gpu used to train the model"
    )

    # Training Basics
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=15000,
        help="maximum epoch number to train",
    )
    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
    )
    # https://blog.csdn.net/qq_43391414/article/details/122992458
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs="+",
        default=[112, 112, 80],
        help="patch size of network input",
    )

    parser.add_argument(
        "--max_samples", type=int, default=80, help="maximum samples to train"
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument("--test_interval_iter", type=int, default=200, help="")
    parser.add_argument("--test_interval_ep", type=int, default=1, help="")
    parser.add_argument("--save_interval_epoch", type=int, default=1000000, help="")
    parser.add_argument(
        "-p",
        "--poly",
        default=False,
        action="store_true",
        help="whether poly scheduler",
    )

    # label and unlabel
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
    parser.add_argument(
        "--labeled_bs", type=int, default=2, help="labeled_batch_size per gpu"
    )
    parser.add_argument("--labeled_num", type=int, default=4, help="labeled data")

    # model related
    parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")

    # unlabeled loss
    parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
    parser.add_argument(
        "--consistency_rampup", type=float, default=40.0, help="consistency_rampup"
    )
    parser.add_argument("--alpha", type=float, default=1.0)

    # parse args
    args = parser.parse_args()
    args = vars(args)

    # 2. update from the config files
    cfgs_file = args["cfg"]
    cfgs_file = os.path.join("./cfgs", cfgs_file)
    with open(cfgs_file, "r") as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # convert "1e-x" to float
    for each in options_yaml.keys():
        tmp_var = options_yaml[each]
        if type(tmp_var) == str and "1e-" in tmp_var:
            options_yaml[each] = float(tmp_var)
    # update original parameters of argparse
    update_values(options_yaml, args)
    import pprint

    # 3. setup gpus and randomness
    # if args["gpu_id"] in range(8):
    if args["gpu_id"] in range(10):
        gid = args["gpu_id"]
    else:
        gid = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)

    if not args["deterministic"]:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args["seed"] > 0:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])

    # 4. outputs and logger
    # 4. outputs and logger
    snapshot_path = "{}/{}_{}_labeled/{}".format(
        args["res_path"],
        args["exp"],
        args["labeled_num"],
        args["model1"] + "_" + args["model2"],
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)
