import argparse
import os
import shutil
import json

import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import logging

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from dataloaders.CAMUS_labeled import BaseDataSets
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import cv2

from configs.config_test import args as FLAGS

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
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    # h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    # image = h5f['image'][:]
    # label = h5f['label'][:]
    image, label = case['image'].unsqueeze(0).cuda(), case['label'].cuda()
    label = label.squeeze(0).cpu().detach().numpy()
    name = case['name'][0]

    # prediction = np.zeros_like(label)

    net.eval()
    test_out = net(image)

    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(image), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        prediction = out

    # for ind in range(image.shape[0]):
    #     slice = image[ind, :, :]
    #     x, y = slice.shape[0], slice.shape[1]
    #     slice = zoom(slice, (256 / x, 256 / y), order=0)
    #     input = torch.from_numpy(slice).unsqueeze(
    #         0).unsqueeze(0).float().cuda()
    #     net.eval()
    #     with torch.no_grad():
    #         if FLAGS.output == "unet_urds":
    #             out_main, _, _, _ = net(input)
    #         else:
    #             out_main = net(input)
    #         out = torch.argmax(torch.softmax(
    #             out_main, dim=1), dim=1).squeeze(0)
    #         out = out.cpu().detach().numpy()
    #         pred = zoom(out, (x / 256, y / 256), order=0)
    #         prediction[ind] = pred
    # 需要修改成单分类

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    image = image.squeeze(0).cpu().detach().numpy()
    image =  np.concatenate([image] * 3, axis=0)

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


def Inference(FLAGS):
    # with open(FLAGS.root_path + '/test.list', 'r') as f:
    #     image_list = f.readlines()
    # image_list = sorted([item.replace('\n', '').split(".")[0]
    #                      for item in image_list])

    name_list_path = os.path.join(FLAGS.root_path, "name.json")
    frame_split_path = os.path.join(FLAGS.root_path, "frame_split_{}.json".format(FLAGS.labeled_num))
    print(FLAGS.labeled_num)
    with open(name_list_path, 'r') as fn:
        name_dict = json.load(fn)['file_name']
    with open(frame_split_path, 'r') as ff:
        split_json = json.load(ff)

    test_idx = split_json['test_idx']
    image_list = [name_dict[i] for i in test_idx]

    # only for fully supervised output

    snapshot_path = "../output/{}_{}patients_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../output/{}_{}patients_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)

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
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    from torch.utils.flop_counter import FlopCounterMode
    inp = torch.randn(1, 1, FLAGS.patch_size[0], FLAGS.patch_size[1]).cuda()
    flop_counter = FlopCounterMode(mods=net, display=False, depth=None)
    with flop_counter:
        net(inp)
    total_flops =  flop_counter.get_total_flops()
    print(total_flops)

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    db_test = BaseDataSets(base_dir=FLAGS.root_path, split='test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(testloader)
    print (len(testloader))



    for i, data in enumerate(pbar):

        first_metric = test_single_volume(data, net, test_save_path, FLAGS)

        first_total += np.asarray(first_metric)


    # for case in tqdm(image_list):
    #     first_metric = test_single_volume(
    #         case, net, test_save_path, FLAGS)
    #     first_total += np.asarray(first_metric)
    #     second_total += np.asarray(second_metric)
    #     third_total += np.asarray(third_metric)
    avg_metric = first_total / len(image_list)

    logger.info("Dice: {}, HD95: {}, ASD: {}".format(avg_metric[0], avg_metric[1], avg_metric[2]) )
    return avg_metric


if __name__ == '__main__':
    # FLAGS = parser.parse_args()
    metric = Inference(FLAGS)

    print(metric)

