import argparse
import logging
import os
import random
import shutil
import sys
import time
import importlib

from torch.cuda import device

import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.optim import lr_scheduler
from dataloaders import utils


from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val import test_single_volume

from configs.config_supervised import args
# from configs.config_supervised_SCSegamba_for_Deepcrack_test import args
# from configs.config_supervised_deepcrack_test import args

# for debug
# from configs.config_supervised_for_debug import args

datasets = ("CRACK500", "DeepCrack")

try:
    import_dataset_name = "dataloaders."+args.dataset+"_labeled"
    dataset_py = importlib.import_module(import_dataset_name)
    BaseDataSets = getattr(dataset_py, "BaseDataSets")

except ImportError:
    print(114514)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def train(args, snapshot_path):
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=3,
                            class_num=num_classes)
        # model, _ = build(args)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # model = model.cpu()
    # crackformer 输出一个元组，末尾的元素是最终分割结果，其余的是每一层的分割图
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    #
    # inp = torch.randn(10, 3, 256, 256).cuda()
    # out = model(inp)
    # print(out.shape)

    # db_train = LabeledDatasets(base_dir=args.data_path, split="train", num=None, transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)]))
    db_train = BaseDataSets(base_dir=args.data_path, split="train", transform="weak")
    # db_val = BaseDataSets(base_dir=args.data_path, split="val", transform=None)
    if args.dataset == 'DeepCrack':
        db_val = BaseDataSets(base_dir=args.data_path, split="train", transform=None)
    else:
        db_val = BaseDataSets(base_dir=args.data_path, split="val", transform=None)

    trainloader = DataLoader(db_train, batch_size = args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    # optimizer = optim.SGD(output.parameters(), lr=base_lr,
    #                       momentum=0.9, weight_decay=0.0001)
    # crackmer
    # optimizer = optim.Adam(model.parameters(), lr=base_lr)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2)


    ce_loss = CrossEntropyLoss()
    bce_loss = BCEWithLogitsLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logger.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    a_dice = args.loss_weight[1]
    a_bce = args.loss_weight[0]
    max_epoch = args.epoch_num
    total_iterations = max_epoch * len(trainloader)
    print(total_iterations)

    model.train()
    # iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in range(max_epoch):

        tbar = tqdm(trainloader, desc='epoch {}'.format(epoch_num))
        loss_list = []
        ce_loss_list = []
        dice_loss_list = []

        for i_batch, sampled_batch in enumerate(tbar):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label'].squeeze(1)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)

            if isinstance(outputs, tuple):
                outputs = outputs[0]
                feature_maps = outputs[1:]

            # outputs_soft = torch.sigmoid(outputs)
            loss_bce = bce_loss(outputs, label_batch.unsqueeze(1))
            outputs_soft = torch.sigmoid(outputs)

            loss_dice = dice_loss(
                outputs_soft, label_batch.unsqueeze(1))

            dice_loss_list.append(loss_dice.item())
            ce_loss_list.append(loss_bce.item())

            supervised_loss = a_dice * loss_dice + a_bce * loss_bce

            loss = supervised_loss
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / total_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # scheduler.step()

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_bce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # logging.info(
            #     'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
            #     (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            # iterator.set_postfix_str('iteration {} : loss : {:.4f}, loss_ce: {:.4f}, loss_dice: {:.4f}'
            #                          .format(iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # if iter_num % 20 == 0:
            #     image = volume_batch[1, 0:1, :, :]
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(
            #         outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction',
            #                      outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % (len(trainloader)) == 0:

                model.eval()
                metric_list = []
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch, model)
                    metric_list.append((metric_i))

                # metric_list = metric_list / len(db_val)
                metric_list = np.array(metric_list)
                # mean_metric = np.mean(metric_list, axis=0)

                # dc, mIoU, p, r, f1

                val_dice = np.mean(metric_list, axis=0)[0]
                val_mIoU = np.mean(metric_list, axis=0)[1]

                # print (performance, iter_num)
                writer.add_scalar('info/val_mean_dice', val_dice, iter_num)
                writer.add_scalar('info/val_mean_IoU', val_mIoU, iter_num)

                if val_mIoU > best_performance:
                    best_performance = val_mIoU
                    save_mode_path = os.path.join(snapshot_path,
                                                  'best_ep_{}_iter_{}_mIoU_{}.pth'.format(
                                                      epoch_num, iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    logger.info("best found at epoch {}".format(epoch_num))
                    # print ("ce loss:", np.mean(ce_loss_list), ", best mIoU:", val_mIoU)

                logger.info(
                    'epoch %d, iteration %d, loss : %f, mean_dice : %f, mean_mIoU : %f' % (epoch_num, iter_num, np.mean(loss_list), val_dice, val_mIoU))
                # print (np.mean(ce_loss_list), np.mean(dice_loss_list))
                model.train()


            if iter_num >= max_iterations:
                break

        # logging.warning(' ep %d : loss : %f, loss_ce: %f, loss_dice: %f' %(epoch_num, tot_loss/len(trainloader), tot_ce_loss/len(trainloader), tot_dice_loss/len(trainloader)))
        print ("ce loss:", np.mean(ce_loss_list), ", mIoU:", val_mIoU)
        if epoch_num !=0 and epoch_num % 10 == 0:
            save_mode_path = os.path.join(
                snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logger.info("save output to {}".format(save_mode_path))

    # print (iter_num)
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../output/{}/{}".format(
        args.exp, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    else:
        for filename in os.listdir(snapshot_path):
            full_path = os.path.join(snapshot_path, filename)
            if os.path.isfile(full_path) and filename.endswith(".pth"):
                os.remove(full_path)


    if not os.path.exists(os.path.join(snapshot_path , 'log') ):
        os.makedirs(os.path.join(snapshot_path , 'log') )

    # if os.path.exists(os.path.join(snapshot_path , 'log.txt')):
    #     os.remove(os.path.join(snapshot_path , 'log.txt'))



    # logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.DEBUG,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))

    global logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(snapshot_path , 'log.txt'))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(str(args))

    train(args, snapshot_path)
