#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   EBA.py
@Time    :   2025/02/11 17:52:50
@Author  :   biabuluo
@Version :   1.0
@Desc    :   Entropy-Based Selection
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from skimage.measure import label
import numpy as np


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=1):
    # probs = F.softmax(output, dim=1)
    _, probs = torch.max(output, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs


def get_LA_masks(out, thres=0.5, nms=1):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_LA(masks)
    return masks


def LargestCC_LA(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def get_pan_mask(out, thres=0.5, nms=True, connect_mode=2):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == True:
        masks = LargestCC_pancreas(masks, connect_mode=connect_mode)
    return masks


def LargestCC_pancreas(segmentation, connect_mode=1):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob, connectivity=connect_mode)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def weighted_mse_loss(output, target, mu=1.0, b=0.5, is_wights=True):
    # 计算MSE损失
    mse_loss = (output - target) ** 2
    # target_soft = torch.softmax(target, dim=1)
    entropy_map = calculate_entropy(target)
    # 根据熵调整权重，熵高的像素点低损失，熵低的像素点高损失
    # 使用熵的反比作为权重，可以通过公式：w = exp(-alpha * entropy_map)
    weights = torch.exp(-mu * entropy_map.unsqueeze(1)) + b  # 在类别维度上扩展
    if is_wights:
        mse_loss = mse_loss * weights
    # 对每个像素求平均损失
    loss = mse_loss.mean()
    return loss


def weighted_ce_loss(
    output_logit,
    target_soft,
    entropy_map,
    mu=1.0,
    b=0.5,
    is_weights=True,
    mode="acdc",
):
    weights = torch.exp(-mu * entropy_map.unsqueeze(1)) + b
    if mode == "acdc":
        target = get_ACDC_masks(target_soft)
    elif mode == "la":
        target = get_LA_masks(target_soft)
    else:
        target = get_pan_mask(target_soft)
    ce_loss = F.cross_entropy(output_logit, target.long(), reduction="none")
    if is_weights:
        ce_loss = ce_loss * weights
    loss = ce_loss.mean()
    return loss


def ebs(x1_soft, x2_soft):
    x1_entropy = calculate_entropy(x1_soft)
    x2_entropy = calculate_entropy(x2_soft)
    pesudo_output, pesedo_encropy = select_output_by_entropy(
        x1_soft, x2_soft, x1_entropy, x2_entropy
    )
    return pesudo_output, pesedo_encropy


def calculate_entropy(softmax_output, dim=1):
    entropy = -torch.sum(softmax_output * torch.log(softmax_output + 1e-10), dim=dim)
    return entropy


def select_output_by_entropy(output1, output2, entropy_map1, entropy_map2):
    selected_output = torch.where(
        entropy_map1.unsqueeze(1) < entropy_map2.unsqueeze(1), output1, output2
    )
    selected_encropy = calculate_entropy(selected_output)
    return selected_output, selected_encropy


if __name__ == "__main__":
    a = torch.rand((4, 3, 112, 112, 80))
    b = torch.rand((4, 3, 112, 112, 80))
    c, d = ebs(a, b)
    print(c.shape, d.shape)
