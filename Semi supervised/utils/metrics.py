#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm


import numpy as np
from medpy import metric


def calculate_metric_percase_val(pred, gt):
    # fix threshold to 0.5

    pred = (pred > 0.5).astype(np.float64)

    if pred.sum() <= 0:
        return 0, 0, 0, 0, 0

    dc = metric.binary.dc(pred, gt)
    mIoU = cal_mIoU_metric(pred, gt)
    p, r, f1 = cal_p_r_f1(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    # hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)

    return dc, mIoU, p, r, f1  #jc, hd, asd

def get_statistics(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    tn = np.sum((pred==0)&(gt==0))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))

    return [tp, tn, fp, fn]


def calculate_metric_percase(pred, gt, threshold_step = 0.01):
    # b = 1, per image

    dc_list = []
    mIoU_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for thresh in np.arange(0.0, 1.0, threshold_step):
        pred_t = (pred > thresh).astype(np.float64)

        if pred_t.sum() <= 0:
            dc = 0
            mIoU = 0
            p = 0
            r = 0
            f1 = 0

        else:
            dc = metric.binary.dc(pred_t, gt)
            mIoU = cal_mIoU_metric(pred_t, gt)
            # TP, TN, FP, FN = get_statistics(pred, gt)
            p, r, f1 = cal_p_r_f1(pred_t, gt)

        dc_list.append(dc)
        mIoU_list.append(mIoU)
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)
        # final_acc_list.append([thresh, dc, mIoU, TP, TN, FP, FN])

    dc_per_img = np.max(np.array(dc_list))
    mIoU_per_img = np.max(np.array(mIoU_list))
    precision_per_img = np.max(np.array(precision_list))
    recall_per_img = np.max(np.array(recall_list))
    f1_per_img = np.max(np.array(f1_list))

    # ois = np.max(f1_list)

    return [dc_per_img, mIoU_per_img, precision_per_img,recall_per_img, f1_per_img]

def cal_p_r_f1(pred, gt):

    TP, TN, FP, FN = get_statistics(pred, gt)
    precision = 1.0 if TP == 0 and FP == 0 else TP / (TP + FP)
    recall = 0 if (TP + FN) <= 0 else TP / (TP + FN)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def cal_mIoU_metric(pred, gt):

    TP, TN, FP, FN = get_statistics(pred, gt)
    # mIoU
    if (FN + FP + TP) <= 0 or (FN + FP + TN) <= 0:
        mIoU = 0
    else:
        iou_1 = TP / (FN + FP + TP)
        iou_0 = TN / (FN + FP + TN)
        mIoU = (iou_1 + iou_0)/2

    return mIoU

def cal_prf_metrics_all(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = gt
            pred_img = (pred > thresh).astype(np.float64)
            statistics.append(get_statistics(pred_img, gt_img))
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[2] for v in statistics])
        fn = np.sum([v[3] for v in statistics])

        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        r_acc = tp / (tp + fn)
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])

    return final_accuracy_all

def cal_mIoU_metric_all(pred_list, gt_list, thresh_step=0.01):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = gt
            pred_img = (pred > thresh).astype(np.float64)

            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0)/2
            iou_list.append(iou)
        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    mIoU = np.max(np.array(final_iou))
    max_threshold = (np.argmax(np.array(final_iou)) + 1) * thresh_step
    return mIoU, max_threshold


def cal_ODS_metric(pred_list, gt_list, thresh_step=0.01):
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = gt
            pred_img = (pred > thresh).astype(np.float64)

            tp, tn, fp, fn = get_statistics(pred_img, gt_img)
            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            ODS_list.append(F1)

        ave_F1 = np.mean(np.array(ODS_list))
        final_ODS.append(ave_F1)
    ODS = np.max(np.array(final_ODS))
    return ODS

def cal_OIS_metric(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        p_acc_list = []
        r_acc_list = []
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = gt
            pred_img = (pred > thresh).astype(np.float64)

            tp, tn, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)

            p_acc_list.append(p_acc)
            r_acc_list.append(r_acc)
            F1_list.append(F1)

        assert len(p_acc_list)==100, "p_acc_list is not 100"
        assert len(r_acc_list)==100, "r_acc_list is not 100"
        assert len(F1_list)==100, "F1_list is not 100"

        max_F1 = np.max(np.array(F1_list))
        final_F1_list.append(max_F1)

    final_F1 = np.sum(np.array(final_F1_list))/len(final_F1_list)
    return final_F1


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
