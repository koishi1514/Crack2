#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm


import numpy as np
from medpy import metric


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float64)
        label_tmp = label_tmp.astype(np.float64)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice

def calculate_metric_percase_val(pred, gt):
    pred[pred > 0.5] = 1
    gt[gt > 0.5] = 1

    if pred.sum() <= 0:
        return 0, 0, 0, 0, 0

    dc = metric.binary.dc(pred, gt)
    mIoU, p, r, f1 = cal_mIoU_metrics(pred, gt)
    return dc, mIoU, p, r, f1  #jc, hd, asd

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() <= 0:
        return 0, 0, 0, 0, 0

    dc = metric.binary.dc(pred, gt)
    mIoU, p, r, f1 = cal_mIoU_metrics(pred, gt)


    # jc = metric.binary.jc(pred, gt)
    # hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)

    return dc, mIoU, p, r, f1  #jc, hd, asd


    # final_F1_list = []
    # for pred, gt in zip(pred_list, gt_list):
    #     p_acc_list = []
    #     r_acc_list = []
    #     F1_list = []
    #     for thresh in np.arange(0.0, 1.0, thresh_step):
    #         gt_img = (gt / 255).astype('uint8')
    #         pred_img = (pred / 255 > thresh).astype('uint8')
    #         tp, fp, fn = get_statistics(pred_img, gt_img)
    #         p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    #         if tp + fn == 0:
    #             r_acc=0
    #         else:
    #             r_acc = tp / (tp + fn)
    #         if p_acc + r_acc==0:
    #             F1 = 0
    #         else:
    #             F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
    #
    #         p_acc_list.append(p_acc)
    #         r_acc_list.append(r_acc)
    #         F1_list.append(F1)
    #
    #     assert len(p_acc_list)==100, "p_acc_list is not 100"
    #     assert len(r_acc_list)==100, "r_acc_list is not 100"
    #     assert len(F1_list)==100, "F1_list is not 100"
    #
    #     max_F1 = np.max(np.array(F1_list))
    #     final_F1_list.append(max_F1)
    #
    # final_F1 = np.sum(np.array(final_F1_list))/len(final_F1_list)
    # return final_F1

def cal_mIoU_metrics(pred, gt):

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    if (FN + FP + TP) <= 0:
        mIoU = 0
    else:
        iou_1 = TP / (FN + FP + TP)
        iou_0 = TN / (FN + FP + TN)
        mIoU = (iou_1 + iou_0)/2

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return mIoU, precision, recall, f1

# def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01, pred_imgs_names=None, gt_imgs_names=None):
    # 源于 scsegamba
    # final_iou = []
    # for thresh in np.arange(0.0, 1.0, thresh_step):
    #     iou_list = []
    #     for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
    #         gt_img = (gt / 255).astype('uint8')
    #     pred_img = (pred / 255 > thresh).astype('uint8')
    #     TP = np.sum((pred_img == 1) & (gt_img == 1))
    #     TN = np.sum((pred_img == 0) & (gt_img == 0))
    #     FP = np.sum((pred_img == 1) & (gt_img == 0))
    #     FN = np.sum((pred_img == 0) & (gt_img == 1))
    #
    #     if (FN + FP + TP) <= 0:
    #         iou = 0
    #     else:
    #         iou_1 = TP / (FN + FP + TP)
    #         iou_0 = TN / (FN + FP + TN)
    #         iou = (iou_1 + iou_0)/2
    #         iou_list.append(iou)
    #         ave_iou = np.mean(np.array(iou_list))
    #         final_iou.append(ave_iou)
    #         mIoU = np.max(np.array(final_iou))
    #
    # return mIoU
