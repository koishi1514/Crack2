import argparse
import os
import shutil

from Demos.RegCreateKeyTransacted import trans

import json

import numpy as np
import torch
from medpy import metric
from tqdm import tqdm
import logging

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from dataloaders.Crack500_labeled import BaseDataSets
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import cv2

from configs.config_supervised_test import args

def draw_sem_seg_by_cv2_sum(image, gt_sem_seg, pred_sem_seg, palette):
    '''
        image: [3,h,w] numpy.ndarray
        gt_sem_seg: [h,w] numpy.ndarray
        pred_sem_seg: [h,w] numpy.ndarray
        palette: [bg, gt, pred, overlap] numpy.ndarray
    '''
    gt_sem_seg = gt_sem_seg.astype(np.uint8)
    pred_sem_seg = pred_sem_seg.astype(np.uint8)
    a = np.unique(gt_sem_seg)
    b = np.unique(pred_sem_seg)
    mask = 2 * pred_sem_seg + gt_sem_seg
    mask = mask.squeeze()

    ids = np.unique(mask)
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
    # h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    # image = h5f['image'][:]
    # label = h5f['label'][:]
    # image, label = case['image'].unsqueeze(0).cuda(), case['label'].cuda()
    # label = label.squeeze(0).cpu().detach().numpy()

    image = case['image'].cuda()
    label = case['label'].cpu().detach().numpy()
    name = case['name'][0]

    # prediction = np.zeros_like(label)

    net.eval()
    test_out = net(image)

    with torch.no_grad():
        out = net(image)
        if isinstance(out, tuple):
            out = out[0]

        out_soft = torch.softmax(out, dim=1)
        out = torch.argmax(out_soft, dim=1)

        out = out.cpu().detach().numpy()
        prediction = out

    prediction = prediction.squeeze()
    label = label.squeeze()

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    image = image.squeeze(0).cpu().detach().numpy()
    # image =  np.concatenate([image] * 3, axis=0)

    palette = [[255, 255, 255],[37, 143, 36], [178, 48, 0], [178, 151, 0]]

    image = draw_sem_seg_by_cv2_sum(image, label, prediction, palette)
    image = cv2.cvtColor(image.transpose(1,2,0), cv2.COLOR_RGB2BGR)
    out_dir =  os.path.join(test_save_path, name+'.png')
    cv2.imwrite(out_dir, image)

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + name + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + name + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + name + "_gt.nii.gz")
    return first_metric


def Inference(args):
    # with open(FLAGS.root_path + '/test.list', 'r') as f:
    #     image_list = f.readlines()
    # image_list = sorted([item.replace('\n', '').split(".")[0]
    #                      for item in image_list])

    # name_list_path = os.path.join(FLAGS.root_path, "name.json")
    # frame_split_path = os.path.join(FLAGS.root_path, "frame_split_{}.json".format(FLAGS.labeled_num))
    # print(FLAGS.labeled_num)
    # with open(name_list_path, 'r') as fn:
    #     name_dict = json.load(fn)['file_name']
    # with open(frame_split_path, 'r') as ff:
    #     split_json = json.load(ff)
    #
    # test_idx = split_json['test_idx']
    # image_list = [name_dict[i] for i in test_idx]

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
    net = net_factory(net_type=args.model, in_chns=1,
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



    for i, data in enumerate(pbar):

        first_metric = test_single_volume(data, net, test_save_path, args)

        first_total += np.asarray(first_metric)


    # for case in tqdm(image_list):
    #     first_metric = test_single_volume(
    #         case, net, test_save_path, FLAGS)
    #     first_total += np.asarray(first_metric)
    #     second_total += np.asarray(second_metric)
    #     third_total += np.asarray(third_metric)
    avg_metric = first_total / test_num

    logger.info("Dice: {}, HD95: {}, ASD: {}".format(avg_metric[0], avg_metric[1], avg_metric[2]) )
    return avg_metric


if __name__ == '__main__':
    # FLAGS = parser.parse_args()
    metric = Inference(args)

    print(metric)

