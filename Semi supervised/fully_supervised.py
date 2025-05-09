import argparse
import logging
import os
import random
import shutil
import sys
import time
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
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils

from dataloaders.Crack500_labeled import BaseDataSets

from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val import test_single_volume

from configs.config_supervised_test import args


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # crackformer 输出一个元组，末尾的元素是最终分割结果，其余的是每一层的分割图

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    #
    # inp = torch.randn(1, 3, 512, 512)
    # _, out=output(inp)
    # print(out.shape)

    # db_train = LabeledDatasets(base_dir=args.data_path, split="train", num=None, transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)]))
    db_train = BaseDataSets(base_dir=args.data_path, split="train", transform="weak")
    db_val = BaseDataSets(base_dir=args.data_path, split="val", transform=None)

    trainloader = DataLoader(db_train, batch_size = args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    # optimizer = optim.SGD(output.parameters(), lr=base_lr,
    #                       momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    bce_loss = BCEWithLogitsLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logger.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    max_epoch = args.epoch_num
    total_iterations = max_epoch * len(trainloader)
    print(total_iterations)

    model.train()
    # iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in range(max_epoch):

        tbar = tqdm(trainloader, desc='epoch {}'.format(epoch_num))
        for i_batch, sampled_batch in enumerate(tbar):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label'].squeeze(1)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            _, outputs = model(volume_batch)
            # if args.model == 'Crackformer':
            #     loss_ce = bce_loss(outputs, label_batch)
            # 原始版本模型适用于多分类
            outputs_soft = torch.softmax(outputs, dim=1)
            loss_ce = ce_loss(outputs, label_batch.long())

            loss_dice = dice_loss(
                outputs_soft, label_batch.unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)


            loss = supervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / total_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
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
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logger.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()


            if iter_num >= max_iterations:
                break

        if epoch_num !=0 and epoch_num % 10 == 0:
            save_mode_path = os.path.join(
                snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logger.info("save output to {}".format(save_mode_path))

    print (iter_num)
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
