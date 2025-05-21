import argparse
import os
import shutil

import json

import numpy as np
import torch
from medpy import metric
from tqdm import tqdm
import logging
from utils import metrics

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from dataloaders.Crack500_labeled import BaseDataSets
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import cv2

from configs.config_supervised import args

def draw_sem_seg_by_cv2_sum(image, gt_sem_seg, pred_sem_seg, palette, threshold=0.5):
    '''
        image: [3,h,w] numpy.ndarray
        gt_sem_seg: [h,w] numpy.ndarray
        pred_sem_seg: [h,w] numpy.ndarray
        palette: [bg, gt, pred, overlap] numpy.ndarray
    '''
    image = (image * 255 if np.max(image) <= 1 else image).astype(np.uint8)
    pred_sem_seg = pred_sem_seg > threshold
    # pred_sem_seg = (pred_sem_seg > threshold if np.max(pred_sem_seg) <= 1 else pred_sem_seg).astype(np.uint8)
    # gt_sem_seg = (gt_sem_seg * 255 if np.max(gt_sem_seg) <= 1 else gt_sem_seg).astype(np.uint8)

    gt_sem_seg = gt_sem_seg.astype(np.uint8)
    pred_sem_seg = pred_sem_seg.astype(np.uint8)
    # a = np.unique(gt_sem_seg)
    # b = np.unique(pred_sem_seg)
    mask = 2 * pred_sem_seg + gt_sem_seg
    mask = mask.squeeze()

    ids = np.unique(mask)
    # palette = [[255, 255, 255],[37, 143, 36], [178, 48, 0], [178, 151, 0]]
    # white, green, red, yellow
    # 背景（无色）, gt（绿色）, pred（红色）,重叠部分（黄色）

    # idx
    color_mask = np.zeros_like(image)
    for idx in ids:
        color_mask[0][mask == idx] = palette[idx][0]
        color_mask[1][mask == idx] = palette[idx][1]
        color_mask[2][mask == idx] = palette[idx][2]

    results = cv2.addWeighted(image, 0.2, color_mask, 0.8, 0)
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i,j] !=0:
    #             image[...,i,j] = results[...,i,j]
    mask = np.expand_dims(mask, 0).repeat(3,axis=0) # np
    image[mask != 0] = results[mask != 0]
    return image

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    return dice, 0, 0


def test_single_volume(case, net, test_save_path, args):

    ts = 0.01
    image = case['image'].cuda()
    label = case['label'].cpu().detach().numpy()
    name = case['name'][0]

    net.eval()

    with torch.no_grad():
        out = net(image)
        if isinstance(out, tuple):
            out = out[0]

        out_soft = torch.sigmoid(out)
        # out_soft = out

        prediction = out_soft.cpu().detach().numpy()

    prediction = prediction.squeeze()
    label = label.squeeze()
    # print (np.unique(prediction),np.unique(label))

    # metrics per image
    metric_single_img = metrics.calculate_metric_percase(prediction, label, ts)
    single_pred = prediction
    single_label = label


    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    image_out = image.squeeze(0).cpu().detach().numpy()

    # opencv need input numpy array pixels are 0-255 and 3 channels
    # image =  np.concatenate([image] * 3, axis=0)

    palette = [[255, 255, 255],[37, 143, 36], [178, 48, 0], [178, 151, 0]]

    draw_output = draw_sem_seg_by_cv2_sum(image_out, label, prediction, palette)
    draw_output = cv2.cvtColor(draw_output.transpose(1,2,0), cv2.COLOR_RGB2BGR)
    out_dir =  os.path.join(test_save_path, name[:-4]+'.png')
    cv2.imwrite(out_dir, draw_output)


    return metric_single_img, single_pred, single_label

def Inference(args):

    # only for fully supervised output

    snapshot_path = "../output/{}/{}".format(
        args.exp, args.model)

    test_save_path = "../output/{}/{}_predictions".format(
        args.exp, args.model)

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(snapshot_path , 'log.txt'))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=args.model, in_chns=3,
                      class_num=args.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(args.model))

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    from torch.utils.flop_counter import FlopCounterMode
    inp = torch.randn(1, 3, args.patch_size[0], args.patch_size[1]).cuda()
    flop_counter = FlopCounterMode(mods=net, display=False, depth=None)
    with flop_counter:
        net(inp)
    total_flops =  flop_counter.get_total_flops()
    print(total_flops)

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    db_test = BaseDataSets(base_dir=args.data_path, split='test', transform=None)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    test_num = len(testloader)
    pbar = tqdm(testloader)
    print (len(testloader))

    pred_list = []
    label_list = []

    for i, data in enumerate(pbar):
        first_metric, pred, lable = test_single_volume(data, net, test_save_path, args)
        first_total += np.asarray(first_metric)
        pred_list.append(pred)
        label_list.append(lable)

    avg_metric = first_total / test_num
    final_accuracy_all = metrics.cal_prf_metrics_all(pred_list, label_list)
    final_accuracy_all = np.array(final_accuracy_all)
    Precision_list, Recall_list, F_list = final_accuracy_all[:, 1], final_accuracy_all[:,2], final_accuracy_all[:, 3]
    mIoU_all, max_threshold_indice = metrics.cal_mIoU_metric_all(pred_list, label_list)
    ois = metrics.cal_OIS_metric(pred_list, label_list)
    ods = metrics.cal_ODS_metric(pred_list, label_list)
    precision = np.max(Precision_list)
    recall = np.max(Recall_list)
    f1 = np.max(F_list)

    # list: ( num * 100 * [thresh, dc, mIoU, TP, TN, FP, FN] )
    # all_img_metric = np.array(all_img_list)
    #
    # all_dc = all_img_metric[..., 1]
    # all_mIoU = all_img_metric[..., 2]
    # all_tp = all_img_metric[..., 3]
    # all_fp = all_img_metric[..., 5]
    # all_fn = all_img_metric[..., 6]
    #
    # sum_tp = np.sum(all_tp, axis=0)
    # sum_fp = np.sum(all_fp, axis=0)
    # sum_fn = np.sum(all_fn, axis=0)
    #
    # final_p = 1.0 if sum_tp==0 and sum_fp==0 else sum_tp / (sum_tp + sum_fp)
    # final_r = sum_tp / (sum_tp + sum_fn)
    # final_f1 = 2 * final_p * final_r / (final_p + final_r)
    #
    # img_ave_mIoU = np.mean(all_mIoU, axis = 0)
    # final_mIoU = np.max(img_ave_mIoU)


    ois1 = avg_metric[4]

    logger.info("metric overall dataset: mIoU: {}, OIS: {}, ODS: {}, F1: {}".format(mIoU_all, ois, ods, f1) )
    logger.info("metric avg image:  Dice: {}, mIoU: {}, precision: {}, recall: {}, F1: {}"
                .format(avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3], avg_metric[4]) )
    return avg_metric


if __name__ == '__main__':
    # FLAGS = parser.parse_args()
    metric = Inference(args)

    print(metric)

