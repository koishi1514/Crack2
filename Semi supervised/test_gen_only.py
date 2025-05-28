import argparse
import os
import shutil
import importlib
import csv
import json

import numpy as np
import torch
from medpy import metric
from tqdm import tqdm
import logging

from urllib3.filepost import writer

from dataloaders import for_test_dataset
from utils import metrics

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
# from dataloaders.CRACK500_labeled import BaseDataSets
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import cv2

from configs.config_supervised_test import args


datasets = ("CRACK500", "DeepCrack")

try:
    import_dataset_name = "dataloaders.for_test_dataset"
    dataset_py = importlib.import_module(import_dataset_name)
    BaseDataSets = getattr(dataset_py, "BaseDataSets")

except ImportError:
    print(114514)


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



def test_single_volume(case, net, test_save_path, args):

    ts = 0.01
    image = case['image'].cuda()
    # label = case['label'].cpu().detach().numpy()
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
    # label = label.squeeze()
    label = np.zeros_like(prediction)
    # print (np.unique(prediction),np.unique(label))

    # metrics per image
    # [dc, mIoU, p, r, f1]


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

    return 0

def Inference(args):

    # only for fully supervised output

    snapshot_path = "../output/{}/{}".format(
        args.exp, args.model)

    test_save_path = "../output/{}/{}_predictions/{}".format(
        args.exp, args.model, args.dataset)

    # csv_save_path = os.path.join(test_save_path, "output.csv")

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(snapshot_path ,'log.txt'))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


    net = net_factory(net_type=args.model, in_chns=3,
                      class_num=args.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(args.model))

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()


    db_test = BaseDataSets(base_dir=args.data_path, split='test', transform=None)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    test_num = len(testloader)
    pbar = tqdm(testloader)
    print (len(testloader))

    pred_list = []
    label_list = []
    metric_per_img_list = []

    for i, data in enumerate(pbar):
        test_single_volume(data, net, test_save_path, args)
        # [dc, mIoU, p, r, f1]


    return 0


if __name__ == '__main__':
    # FLAGS = parser.parse_args()
    metric = Inference(args)


