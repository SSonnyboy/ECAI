#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   bcp.py
@Time    :   2025/03/26 18:00:13
@Author  :   biabuluo
@Version :   1.0
@Desc    :   https://github.com/DeepMed-Lab-ECNU/BCP/blob/main/code/ACDC_BCP_train.py#L131
"""
import torch
import numpy as np
import torch.nn as nn


def generate_mask_PAN(img, patch_size=64):
    batch_l = img.shape[0]
    # batch_unlab = unimg.shape[0]
    loss_mask = torch.ones(batch_l // 2, 96, 96, 96).cuda()
    # loss_mask_unlab = torch.ones(batch_unlab, 96, 96, 96).cuda()
    mask = torch.ones(96, 96, 96).cuda()
    w = np.random.randint(0, 96 - patch_size)
    h = np.random.randint(0, 96 - patch_size)
    z = np.random.randint(0, 96 - patch_size)
    mask[w : w + patch_size, h : h + patch_size, z : z + patch_size] = 0
    loss_mask[:, w : w + patch_size, h : h + patch_size, z : z + patch_size] = 0
    # loss_mask_unlab[:, w:w+patch_size, h:h+patch_size, z:z+patch_size] = 0
    # cordi = [w, h, z]
    return mask.long(), loss_mask.long()


def generate_mask_LA(img, mask_ratio=2 / 3):
    batch_size, channel, img_x, img_y, img_z = (
        img.shape[0],
        img.shape[1],
        img.shape[2],
        img.shape[3],
        img.shape[4],
    )
    loss_mask = torch.ones(batch_size // 2, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = (
        int(img_x * mask_ratio),
        int(img_y * mask_ratio),
        int(img_z * mask_ratio),
    )
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w : w + patch_pixel_x, h : h + patch_pixel_y, z : z + patch_pixel_z] = 0
    loss_mask[
        :, w : w + patch_pixel_x, h : h + patch_pixel_y, z : z + patch_pixel_z
    ] = 0
    return mask.long(), loss_mask.long()


def generate_mask_ACDC(img_lb):
    batch_size, channel, img_x, img_y = (
        img_lb.shape[0],
        img_lb.shape[1],
        img_lb.shape[2],
        img_lb.shape[3],
    )
    loss_mask = torch.ones(batch_size // 2, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w : w + patch_x, h : h + patch_y] = 0  # 中间0 四周1
    loss_mask[:, w : w + patch_x, h : h + patch_y] = 0  # 中间0 四周1
    return mask.long(), loss_mask.long()


def copy_paste(labeled_batch, y_batch, unlabeled_batch, pseudo_batch, img_mask):
    uimg_a, uimg_b = unlabeled_batch.chunk(2)
    img_a, img_b = labeled_batch.chunk(2)

    ulab_a, ulab_b = pseudo_batch.chunk(2)
    lab_a, lab_b = y_batch.chunk(2)

    net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
    net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)

    unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
    l_label = lab_b * img_mask + ulab_b * (1 - img_mask)

    return net_input_unl, net_input_l, unl_label, l_label


def to_one_hot(tensor, nClasses):
    """Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, "one hot tensor.max() = {} < {}".format(
        torch.max(tensor), nClasses
    )
    assert tensor.min().item() >= 0, "one hot tensor.min() = {} < {}".format(
        tensor.min(), 0
    )

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


import torch.nn.functional as F


def get_probability(logits):
    """Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


class DiceLoss_bcp_ACDC(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_bcp_ACDC, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), "predict & target shape do not match"
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(
                torch.ones((1, nclass)).type(torch.float32), requires_grad=False
            )
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(
                torch.tensor(class_weights, dtype=torch.float32), requires_grad=False
            )

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]
        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)

            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction="none")


def mix_loss_3d(
    net3_output,
    img_l,
    patch_l,
    mask,
    weight_map_ul,
    l_weight=1.0,
    u_weight=0.5,
    unlab=False,
):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    weight_map_l = torch.ones_like(weight_map_ul)
    img_map, patch_map = weight_map_l, weight_map_ul
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
        img_map, patch_map = weight_map_ul, weight_map_l
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask * img_map) * image_weight
    dice_loss += DICE(net3_output, patch_l, patch_mask * patch_map) * patch_weight
    loss_ce = (
        image_weight
        * (CE(net3_output, img_l) * mask * img_map).sum()
        / (mask.sum() + 1e-16)
    )
    loss_ce += (
        patch_weight
        * (CE(net3_output, patch_l) * patch_mask * patch_map).sum()
        / (patch_mask.sum() + 1e-16)
    )
    loss = (dice_loss + loss_ce) / 2
    return loss
