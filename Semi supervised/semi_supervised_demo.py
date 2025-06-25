# from dataset.semi import SemiDataset
# from output.semseg.deeplabv2 import DeepLabV2
# from output.semseg.deeplabv3plus import DeepLabV3Plus
# from output.semseg.pspnet import PSPNet
# from utils import count_params, meanIOU, color_map

import argparse
from copy import deepcopy
import numpy as np
import os
import shutil
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import math

"---------------------------------------------------"
import json
from networks.net_factory import net_factory
from torchvision import transforms
from configs.config_st_plus_plus_test import args
# from new_train_mean_teacher_2D import train

from utils import losses, metrics, ramps
# from dataloaders.CAMUS import (BaseDataSets, RandomGenerator,
#                                TwoStreamBatchSampler)
# 可能后续需要添加新的dataloader 做半监督训练
from dataloaders.CAMUS_labeled import BaseDataSets as LabeledDatasets
from dataloaders.CAMUS_labeled import RandomGenerator
from dataloaders.CAMUS_unlabeled import BaseDataSets as UnlabeledDatasets
# from utils.losses import conf_ce_loss

from medpy import metric

MODE = None

def create_model(ema=False):
    # Network definition
    model = net_factory(net_type=args.model, in_chns=1,
                        class_num=args.num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
            # param.requires_grad = False
    return model

def init_basic_elems(args):
    # model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    # output = model_zoo[args.output](args.backbone, 21 if args.dataset == 'pascal' else 19)
    model = create_model(ema=False)

    # optimizer = SGD(output.parameters(), lr=args.base_lr,
    #                 momentum=0.9, weight_decay=0.0001)
    optimizer = AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.0001)

    head_lr_multiple = 1.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    # optimizer = SGD([{'params': output.backbone.parameters(), 'lr': args.lr},
    #                  {'params': [param for name, param in output.named_parameters()
    #                              if 'backbone' not in name],
    #                   'lr': args.lr * head_lr_multiple}],
    #                 lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()
    # output = output.cuda()

    return model, optimizer

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_param_momentum(pm, current_train_iter, total_iters):
    return 1.0 - (1.0 - pm) * (
            (math.cos(math.pi * current_train_iter / total_iters) + 1) * 0.5
    )

def update_momentum(model, ema_model, m, perserving_rate):
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        # p2.data = m * p2.data + (1.0 - m) * p1.detach().data
        tmp_prob = np.random.rand()
        if tmp_prob < perserving_rate:
            pass
        else:
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data

def assist_mask_calculate(args, core_predict, assist_predict, topk=1):
    _, index = torch.topk(assist_predict, k=topk, dim=1)
    mask = torch.nn.functional.one_hot(index.squeeze())
    # k!= 1, sum them
    mask = mask.sum(dim=1) if topk > 1 else mask
    if mask.shape[-1] != args.num_classes:
        expand = torch.zeros(
            [mask.shape[0], mask.shape[1], mask.shape[2], args.num_classes - mask.shape[-1]]).cuda()
        mask = torch.cat((mask, expand), dim=3)
    mask = mask.permute(0, 3, 1, 2)
    # get the topk result of the assist map
    assist_predict = torch.mul(assist_predict, mask)

    # fullfill with core predict value for the other entries;
    # as it will be merged based on threshold value
    assist_predict[torch.where(assist_predict == .0)] = core_predict[torch.where(assist_predict == .0)]
    return assist_predict

def main(args, snapshot_path):


    if not os.path.exists(os.path.join(args.root_path, "PseudoMask") ):
        os.makedirs(os.path.join(args.root_path, "PseudoMask") )
    else:
        shutil.rmtree(os.path.join(args.root_path, "PseudoMask"))
        os.makedirs(os.path.join(args.root_path, "PseudoMask"))


    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    stage = 1


    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    valset = LabeledDatasets(base_dir=args.root_path, split="val")
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)


    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))
    logger.info('\n================> Total stage %d/%i: '
                'Supervised training on labeled images (SupOnly)' % (stage, 6 if args.plus else 3))

    global MODE
    MODE = 'train'


    trainset_labeled = LabeledDatasets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]), labeled_num=args.labeled_num)
    trainset_unlabeled = UnlabeledDatasets(base_dir=args.root_path, split="train", num=None,
                                       transform=transforms.Compose([RandomGenerator(args.patch_size)]), labeled_num=args.labeled_num)
    trainset_unlabeled2 = torch.utils.data.ConcatDataset([trainset_unlabeled, trainset_unlabeled])

    trainloader_l = DataLoader(trainset_labeled, batch_size=args.labeled_bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    trainloader_ul = DataLoader(trainset_unlabeled, batch_size=args.batch_size-args.labeled_bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)


    # ul_sampler = torch.utils.data.sampler.RandomSampler(trainset_unlabeled, replacement=True, num_samples=len(trainset_labeled))


    model, optimizer = init_basic_elems(args)
    # print('\nParams: %.1fM' % count_params(output))

    # from torch.utils.flop_counter import FlopCounterMode
    # inp = torch.randn(1, 1, args.patch_size[0], args.patch_size[1]).cuda()
    # flop_counter = FlopCounterMode(mods=output, display=False, depth=None)
    # with flop_counter:
    #     output(inp)
    # total_flops =  flop_counter.get_total_flops()
    # print(total_flops)

    best_model, checkpoints, model_t1, model_t2 = train(model, trainloader_l, trainloader_ul, valloader, ce_loss, dice_loss, optimizer, args, stage)
    stage += 1


    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images'.format(stage))
        logger.info('\n\n\n================> Total stage {}/3: Pseudo labeling all unlabeled images'.format(stage))

        # dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)

        labelset = UnlabeledDatasets(base_dir=args.root_path, split="label_all", addition='all', labeled_num=args.labeled_num)
        labelloader = DataLoader(labelset, batch_size=1, shuffle=False, num_workers=0)

        label(best_model, labelloader, args)
        stage += 1

        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images'.format(stage))
        logger.info('\n\n\n================> Total stage {}/3: Re-training on labeled and unlabeled images'.format(stage))

        MODE = 'semi_train'


        trainset_labeled = LabeledDatasets(base_dir=args.root_path, split="retrain", num=None,
                                           transform=transforms.Compose([RandomGenerator(args.patch_size)]), addition='all', labeled_num=args.labeled_num)
        trainset_unlabeled = UnlabeledDatasets(base_dir=args.root_path, split="retrain", num=None,
                                               transform=transforms.Compose([RandomGenerator(args.patch_size)]), labeled_num=args.labeled_num)

        trainloader_l = DataLoader(trainset_labeled, batch_size=args.labeled_bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        trainloader_ul = DataLoader(trainset_unlabeled, batch_size=args.batch_size-args.labeled_bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        trainloader_ul2 = DataLoader(trainset_unlabeled2, batch_size=args.batch_size-args.labeled_bs, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

        model, optimizer = init_basic_elems(args)

        train(model, trainloader_l, trainloader_ul2, valloader, ce_loss, dice_loss, optimizer, args, stage)

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training'.format(stage))
    logger.info('\n\n\n================> Total stage {}/6: Select reliable images for the 1st stage re-training'.format(stage))

    # dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    # labelset = BaseDataSets(base_dir=args.root_path, split="label_all")
    labelset = UnlabeledDatasets(base_dir=args.root_path, split="label_all", num=None, addition='all', labeled_num=args.labeled_num)
    labelloader = DataLoader(labelset, batch_size=1, shuffle=False, num_workers=0)
    logger.info("total unlabeled data: {}".format(len(labelset)))

    # reliable_id_path = os.path.join(snapshot_path, "reliable")
    select_reliable(checkpoints, labelloader, args, model_t1, model_t2, best_model)
    stage += 1

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images'.format(stage))
    logger.info('\n\n\n================> Total stage {}/6: Pseudo labeling reliable images'.format(stage))

    labelset = UnlabeledDatasets(base_dir=args.root_path, split="label_semi", num=None, addition='reliable', labeled_num=args.labeled_num)
    labelloader = DataLoader(labelset, batch_size=1, shuffle=False, num_workers=0)
    logger.info("reliable unlabeled data: {}".format(len(labelset)))

    label(best_model, labelloader, args)
    stage += 1

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')
    logger.info('\n\n\n================> Total stage {}/6: The 1st stage re-training on labeled and reliable unlabeled images'.format(stage))

    MODE = 'semi_train'

    trainset_labeled = LabeledDatasets(base_dir=args.root_path, split="retrain", num=None,
                                       transform=transforms.Compose([RandomGenerator(args.patch_size)]), addition='reliable', labeled_num=args.labeled_num)
    trainset_unlabeled = UnlabeledDatasets(base_dir=args.root_path, split="retrain", num=None,
                                           transform=transforms.Compose([RandomGenerator(args.patch_size)]), labeled_num=args.labeled_num)
    trainset_unlabeled2 = torch.utils.data.ConcatDataset([trainset_unlabeled, trainset_unlabeled])
    logger.info("trainset labeled data: {}, unlabeled data: {}, unlabeled set size: {} "
                .format(len(trainset_labeled), len(trainset_unlabeled), len(trainset_unlabeled2)))

    trainloader_l = DataLoader(trainset_labeled, batch_size=args.labeled_bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    trainloader_ul = DataLoader(trainset_unlabeled2, batch_size=args.batch_size-args.labeled_bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader_l, trainloader_ul, valloader, ce_loss, dice_loss, optimizer, args, stage)
    stage += 1

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')
    logger.info('\n\n\n================> Total stage {}/6: Pseudo labeling unreliable images'.format(stage))

    labelset = UnlabeledDatasets(base_dir=args.root_path, split="label_semi", num=None, addition='unreliable', labeled_num=args.labeled_num)
    labelloader = DataLoader(labelset, batch_size=1, shuffle=False, num_workers=0)
    logger.info("unreliable unlabeled data: {}".format(len(labelset)))

    label(best_model, labelloader, args)
    stage += 1

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')
    logger.info('\n\n\n================> Total stage {}/6: The 2nd stage re-training on labeled and all unlabeled images'.format(stage))

    trainset_labeled = LabeledDatasets(base_dir=args.root_path, split="retrain", num=None,
                                       transform=transforms.Compose([RandomGenerator(args.patch_size)]), addition='all', labeled_num=args.labeled_num)

    logger.info("trainset labeled data: {}, unlabeled data: {}, unlabeled set size: {} "
                .format(len(trainset_labeled), len(trainset_unlabeled), len(trainset_unlabeled2)))

    trainloader_l = DataLoader(trainset_labeled, batch_size=args.labeled_bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    # trainloader_ul = DataLoader(trainset_unlabeled, batch_size=args.batch_size-args.labeled_bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    trainloader_ul2 = DataLoader(trainset_unlabeled2, batch_size=args.batch_size-args.labeled_bs, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader_l, trainloader_ul2, valloader, ce_loss, dice_loss, optimizer, args, stage)


def train(model, trainloader_l, trainloader_ul, valloader, ce_loss, dice_loss, optimizer, args, stage):

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    gamma = 0.5


    iters = 0
    total_iters = len(trainloader_l) * args.epoch_num

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []

    ema_model1 = create_model(ema=True)
    ema_model2 = create_model(ema=True)

    best_epoch = 0
    best_model = model

    for epoch in range(args.epoch_num):

        if epoch % 2==0:
            flag = 1
        else:
            flag = 2

        # print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
        #       (epoch, optimizer.param_groups[0]["lr"], previous_best))

        logger.info('epoch %d : lr : %f best_dice : %f' % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader_l)

        for i, (sampled_batch, sampled_batch_ul) in enumerate(zip(tbar, trainloader_ul)):

            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            unlabeled_volume_batch = sampled_batch_ul['image'].cuda()
            # print(sampled_batch['label'].min(), sampled_batch['label'].max())
            # print(sampled_batch['image'].min(), sampled_batch['image'].max())
            # print(sampled_batch_ul['label'].min(), sampled_batch_ul['label'].max())

            # unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            volume_batch = torch.cat((volume_batch, unlabeled_volume_batch), dim=0)

            # volume_batch SHAPE: [bs, c, h, w]
            # label_batch SHAPE: [labeled_bs, h, w]

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # outputs SHAPE: [b, num_classes, h, w]

            # with torch.no_grad():
            #     ema_output = ema_model(ema_inputs)
            #     ema_output_soft = torch.softmax(ema_output, dim=1)

            with torch.no_grad():
                ema_output1 = ema_model1(ema_inputs)
                ema_output2 = ema_model2(ema_inputs)

                if flag==1:
                    ema_output2 = assist_mask_calculate(args=args, core_predict=ema_output1, assist_predict=ema_output2, topk=1)

                else:
                    ema_output1 = assist_mask_calculate(args=args, core_predict=ema_output2, assist_predict=ema_output1, topk=1)


                # ema_output = torch.softmax(ema_output, dim=1)
                # ema_output_aux = torch.softmax(ema_output_aux, dim=1)

                ema_output = gamma * ema_output1 + (1 - gamma) * ema_output2
                ema_output_soft = torch.softmax(ema_output, dim=1)



            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())


            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = get_current_consistency_weight(epoch)

            if epoch < 5:
                consistency_loss = 0.0
            else:
                # conf_ce_loss(inputs=volume_batch[:args.labeled_bs], targets=ema_output, conf_mask=True, threshold=0.6, threshold_neg=0.6)
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:]-ema_output_soft)**2)

            loss = supervised_loss + consistency_weight * consistency_loss
            total_loss += loss.item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = get_param_momentum(pm = 0.99, current_train_iter = iters, total_iters = total_iters)

            if flag ==1:
                # update_ema_variables(output, ema_model1, args.ema_decay, iters)
                update_momentum(model, ema_model1, m, 0.7)
            if flag ==2:
                # update_ema_variables(output, ema_model2, args.ema_decay, iters)
                update_momentum(model, ema_model2, m, 0.7)

            lr_ = base_lr * (1.0 - iters / total_iters) ** 0.9


            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iters = iters + 1


            # pred = output(img)
            # loss = criterion(pred, mask)
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #
            # total_loss += loss.item()
            #
            # iters += 1
            # lr = args.lr * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * 1.0 if args.output == 'deeplabv2' else lr * 10.0
            # logger.info('epoch %d batch : %f, Loss : %.3f' % (epoch, i, (total_loss / (i + 1) )))

            tbar.set_description('Epoch: {}, Loss: {:.5f}'.format(epoch, total_loss / (i + 1)))

        # metric = meanIOU(num_classes=args.num_classes-1)

        model.eval()
        tbar = tqdm(valloader)
        metric_list = 0.

        with torch.no_grad():
            for b, sampled_batch in enumerate(tbar):

                # CAMUS
                volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

                # volume_batch SHAPE: [1, c, h, w], b=1
                # label_batch SHAPE: [h, w], no batch dimension

                volume_batch = volume_batch.unsqueeze(1)
                label_batch = label_batch.squeeze(0)

                # both [c, h, w], c=1 on CAMUS

                outputs = model(volume_batch)
                outputs_cal = torch.argmax(torch.softmax(outputs, dim=1), dim=1).float().squeeze(0)

                classes_metric = []

                for i in range(1, args.num_classes):
                    classes_metric.append ( calculate_metric_percase(pred = (outputs_cal==i).cpu().numpy(), gt = (label_batch==i).cpu().numpy() ) )


                metric_list += np.array(classes_metric)

                mean_metric = metric_list/(b+1)

                # metric.add_batch(outputs.cpu().numpy(), label_batch.cpu().numpy())

                # img = img.cuda()
                # pred = output(img)
                # pred = torch.argmax(pred, dim=1)

                # metric.add_batch(pred.cpu().numpy(), mask.numpy())
                # mIOU = metric.evaluate()[-1]

                #　临时只考虑一类 mean_metric[i][0] 第i类的dice，mean_metric[i][1] 第i类的hd95
                tbar.set_description( 'Dice: {:.5f} '.format(mean_metric[0][0])+'HD95: {:.4f} '.format(mean_metric[0][1])  )

        mean_dice = mean_metric[0][0]

        logger.info('epoch %d : Dice : %.5f' % (epoch, mean_dice))

        # mIOU *= 100.0
        if mean_dice > previous_best:
            if previous_best != 0:
                # os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.output, args.backbone, previous_best)))
                os.remove(os.path.join(snapshot_path, 'stage{}_epoch{}_dice_{:.4f}.pth'.format (stage, best_epoch, previous_best)))
            previous_best = mean_dice
            best_epoch = epoch
            torch.save(model.module.state_dict(), os.path.join(snapshot_path, 'stage{}_epoch{}_dice_{:.4f}.pth'.format (stage, best_epoch, previous_best)))

            if args.plus:
                if stage == 6:
                    torch.save(model.module.state_dict(), os.path.join(snapshot_path, '{}_best_model.pth'.format (args.model )))
                else:
                    torch.save(model.module.state_dict(), os.path.join(snapshot_path, 'stage{}_{}_best_model.pth'.format (stage, args.model )))
            else:
                if stage == 3:
                    torch.save(model.module.state_dict(), os.path.join(snapshot_path, '{}_best_model.pth'.format (args.model )))
                else:
                    torch.save(model.module.state_dict(), os.path.join(snapshot_path, 'stage{}_{}_best_model.pth'.format (stage, args.model )))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epoch_num // 3, args.epoch_num * 2 // 3, args.epoch_num]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints, ema_model1, ema_model2

    return best_model


def select_reliable(models, dataloader, args, model_t1, model_t2, best):

    reliable_id_path = args.root_path

    if not os.path.exists(reliable_id_path):
        os.makedirs(reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for b, sampled_batch in enumerate(tbar):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            name = sampled_batch['name'][0]
            volume_batch = volume_batch.unsqueeze(1)
            label_batch = label_batch.squeeze(0)

            preds = []
            for model in models:
                output = model(volume_batch)
                preds.append(torch.argmax(torch.softmax(output, dim=1), dim=1).float().squeeze(0).cpu().numpy())

            # mIOU = []
            # for i in range(len(preds) - 1):
            #     metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
            #     metric.add_batch(preds[i], preds[-1])
            #     mIOU.append(metric.evaluate()[-1])
            #
            # reliability = sum(mIOU) / len(mIOU)

            # output_t1 = model_t1(volume_batch)
            # output_t2 = model_t2(volume_batch)
            output_best = best(volume_batch)
            pred_best = torch.argmax(torch.softmax(output_best, dim=1), dim=1).float().squeeze(0).cpu().numpy()
            # pred_t1 = torch.argmax(torch.softmax(output_t1, dim=1), dim=1).float().squeeze(0).cpu().numpy()
            # pred_t2 = torch.argmax(torch.softmax(output_t2, dim=1), dim=1).float().squeeze(0).cpu().numpy()


            dice_list = []
            for i in range(len(preds) - 1):
                dice, hd95 = calculate_metric_percase(pred = (preds[i]==1), gt = (pred_best==1) )
                # dice_2, _ = calculate_metric_percase(pred = (preds[i]==1), gt = (pred_t1==1) )
                # dice_3, _ = calculate_metric_percase(pred = (preds[i]==1), gt = (pred_t2==1) )
                # dice = 0.5 * dice_1 + 0.25 * (dice_2 + dice_3)
                dice_list.append(dice)

            reliability = np.mean(dice_list)

            id_to_reliability.append((name, reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    reliable_dict = {
        "reliable_name": [elem[0] for elem in  id_to_reliability[:len(id_to_reliability) // 2] ],
        "unreliable_name": [elem[0] for elem in  id_to_reliability[len(id_to_reliability) // 2:] ]
    }
    if os.path.exists(os.path.join(reliable_id_path, 'reliable_name.json')):
        os.remove(os.path.join(reliable_id_path, 'reliable_name.json'))

    with open(os.path.join(reliable_id_path, 'reliable_name.json'), 'w') as json_file:
        json.dump(reliable_dict, json_file, indent=4)


    # with open(os.path.join(reliable_id_path, 'reliable_ids.txt'), 'w') as f:
    #     for elem in id_to_reliability[:len(id_to_reliability) // 2]:
    #         f.write(elem[0] + '\n')
    # with open(os.path.join(reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
    #     for elem in id_to_reliability[len(id_to_reliability) // 2:]:
    #         f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)

    # cmap = color_map(args.dataset)

    with torch.no_grad():
        for b, sampled_batch in enumerate(tbar):
            # CAMUS
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            volume_batch = volume_batch.unsqueeze(1)
            label_batch = label_batch.squeeze(0)

            outputs = model(volume_batch) # [1, num_classes, h, w]
            outputs_cal = torch.argmax(torch.softmax(outputs, dim=1), dim=1).float().squeeze(0) # make it [1, h, w]
            pred = outputs_cal.detach().cpu().numpy()

            classes_metric = []
            for i in range(1, args.num_classes):
                classes_metric.append ( calculate_metric_percase(pred = (outputs_cal==i).cpu().numpy(), gt = (label_batch==i).cpu().numpy() ) )

            # 暂时只分割一类
            dice = classes_metric[0][0]
            hd95 = classes_metric[0][1]

            # 这里有个问题 我直接保存成01二值即可？
            # pred = Image.fromarray(outputs_cal.squeeze(0).numpy().astype(np.uint8), mode='P')
            # pred.putpalette(cmap)

            # pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            pseduo_mask_path = os.path.join(args.root_path, "PseudoMask", sampled_batch['name'][0]+ ".npy")
            pseduo_mask_img_path = os.path.join(args.root_path, "PseudoImg", sampled_batch['name'][0]+ ".png")
            np.save (pseduo_mask_path, pred)

            # img = Image.fromarray((pred*255).astype(np.uint8))
            # # Save as PNG
            # img.save(pseduo_mask_img_path)

            tbar.set_description( 'Dice: {:.5f} '.format(dice)+'HD95: {:.4f} '.format(hd95)  )



if __name__ == '__main__':

    if args.epoch_num is None:
        args.epoch_num = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    if args.base_lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    # if args.crop_size is None:
    #     args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]

    snapshot_path = "../output/{}_{}patients_labeled/{}".format(args.exp, args.labeled_num, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(os.path.join(snapshot_path , 'log') ):
        os.makedirs(os.path.join(snapshot_path , 'log') )

    global logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(snapshot_path, 'log.txt'))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(str(args))

    main(args, snapshot_path)
