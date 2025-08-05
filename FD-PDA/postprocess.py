import argparse
import os
import shutil
import importlib
import csv
import json
import random
import copy

import numpy as np
import torch
from medpy import metric
from tqdm import tqdm
import logging

from utils import metrics

from networks.net_factory import net_factory
from dataloaders.create_dataset import NewDataSets
from torch.utils.data.dataloader import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from utils import losses, metrics, ramps, get_densecrf



from PIL import Image
import cv2

# from configs.config_supervised import args
# from configs.config_supervised_SCSegamba_for_Deepcrack_test import args
# from configs.config_supervised_post_training_real_data import args
from configs.config_supervised_post_training import args
from dataloaders.CRACK500_labeled import BaseDataSets as BaseDataSets_origin
from test import draw_sem_seg_by_cv2_sum


datasets = ("CRACK500", "DeepCrack","CrackTree","CFD")

try:
    import_dataset_name = "dataloaders."+args.dataset+"_labeled"
    # import_dataset_name = "dataloaders.real_data_20250704"
    dataset_py = importlib.import_module(import_dataset_name)
    BaseDataSets = getattr(dataset_py, "BaseDataSets")

except ImportError:
    print(114514)


def test_single_volume(case, net, test_save_path=None, args=None):

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
    metric_single_img = metrics.calculate_metric_percase_val(prediction, label)
    # [dc, mIoU, p, r, f1]
    single_pred = prediction
    single_label = label

    return metric_single_img, single_pred, single_label

def Inference(args, save_path, net, iternum, mask_range, need_save=False):

    # only for fully supervised output


    test_save_path = save_path
    # test_save_path_th = "../output/{}/{}_predictions/post_train_th".format(
    #     args.exp, args.model, args.dataset)

    csv_save_path = os.path.join(test_save_path, "output.csv")

    # if os.path.exists(test_save_path):
    #     shutil.rmtree(test_save_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    net.eval()

    first_total = 0.0

    db_test = BaseDataSets(base_dir=args.data_path, split='test', transform=None)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    test_num = len(testloader)
    pbar = tqdm(testloader)
    print (len(testloader))

    image_list = []
    label_list = []
    pred_list = []
    pred_post_list = []
    mask_list = []
    name_list = []

    metric_per_img_list = []
    image_list = []
    metric_header = ["Name", "Dice", "mIoU", "Precision", "Recall", "F1 score"]

    for i, data in enumerate(pbar):

        first_metric, pred, label = test_single_volume(data, net, test_save_path, args)
        # [dc, mIoU, p, r, f1]
        first_total += np.asarray(first_metric)
        first_metric.insert(0, data['name'][0])

        metric_per_img_list.append(first_metric)
        pred_list.append(pred)
        label_list.append(label)

        image = data['image'].cpu().detach().numpy().squeeze(0)
        name = data['name'][0]
        image_list.append(image)
        name_list.append(name)

        # pred_crf = get_densecrf.dense_crf(image, pred)
        # pred_crf_list.append(pred_crf)

        # 改为人工选取阈值（暂时）
        pred_post = pred.copy()
        pred_post[pred_post >= mask_range[1]] = 1
        pred_post[pred_post <= mask_range[0]] = 0
        mask = np.where((pred >= mask_range[1]) | (pred <= mask_range[0]), 1, 0)
        # print(mask.sum())
        mask_list.append(mask)
        pred_post_list.append(pred_post)

        if need_save:
            image_out = image
            palette = [[255, 255, 255],[37, 143, 36], [178, 48, 0], [178, 151, 0]]
            draw_output = draw_sem_seg_by_cv2_sum(image_out, label, pred, palette)
            draw_output = cv2.cvtColor(draw_output.transpose(1,2,0), cv2.COLOR_RGB2BGR)
            out_dir =  os.path.join(test_save_path, name[:-4]+'.png')
            cv2.imwrite(out_dir, draw_output)

            # draw_output = draw_sem_seg_by_cv2_sum(image_out, label, pred_post, palette)
            # draw_output = cv2.cvtColor(draw_output.transpose(1,2,0), cv2.COLOR_RGB2BGR)
            # out_dir =  os.path.join(test_save_path, name[:-4]+'.png')
            # cv2.imwrite(out_dir, draw_output)


    avg_metric = first_total / test_num
    dataset = NewDataSets(img_list=image_list, label_list=pred_post_list, name_list=name_list, mask_list=mask_list, real_label_list = label_list, transform=None, split='train')

    if need_save:
        final_accuracy_all = metrics.cal_prf_metrics_all(pred_list, label_list)
        final_accuracy_all = np.array(final_accuracy_all)
        Precision_list, Recall_list, F_list = final_accuracy_all[:, 1], final_accuracy_all[:,2], final_accuracy_all[:, 3]
        mIoU_all, max_threshold_indice = metrics.cal_mIoU_metric_all(pred_list, label_list)
        ois = metrics.cal_OIS_metric(pred_list, label_list)
        ods = metrics.cal_ODS_metric(pred_list, label_list)
        precision = np.max(Precision_list)
        recall = np.max(Recall_list)
        f1 = np.max(F_list)
        print (max_threshold_indice)

        logger.info("post train iter {}, metric overall dataset: mIoU: {}, OIS: {}, ODS: {}, F1: {}".format(iternum, mIoU_all, ois, ods, f1) )
        # logger.info("metric avg image:  Dice: {}, mIoU: {}, precision: {}, recall: {}, F1: {}"
        #             .format(avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3], avg_metric[4]) )
        print("post train iter {}, metric overall dataset: mIoU: {}, OIS: {}, ODS: {}, F1: {}".format(iternum, mIoU_all, ois, ods, f1))



    if need_save:
        with open(csv_save_path, mode='w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(metric_header)
            writer.writerows(metric_per_img_list)
            writer.writerow([" ", "mIoU", "ois", "ods", "F1 score"])
            writer.writerow(["overall_metrics", mIoU_all, ois, ods, f1])


    # print("metric avg image:  Dice: {}, mIoU: {}, precision: {}, recall: {}, F1: {}"
    #       .format(avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3], avg_metric[4]) )
    return avg_metric, dataset


def post_train(args, snapshot_path, new_dataset, model, best_performance=0.0):

    origin_dataset_path = '../dataset/CRACK500/'
    base_lr = args.base_lr
    print(base_lr)
    weight_decay = args.weight_decay
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # model = model.cpu()
    # crackformer 输出一个元组，末尾的元素是最终分割结果，其余的是每一层的分割图
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train_origin = BaseDataSets_origin(base_dir=origin_dataset_path, split="train", transform="weak")
    db_train = new_dataset
    db_val = BaseDataSets(base_dir=args.data_path, split='test', transform=None)
    # if args.dataset == 'DeepCrack':
    #     db_val = BaseDataSets(base_dir=args.data_path, split="train", transform=None)
    # else:
    #     db_val = BaseDataSets(base_dir=args.data_path, split="val", transform=None)
    trainloader_origin = DataLoader(db_train_origin, batch_size = args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    trainloader = DataLoader(db_train, batch_size = args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    # optimizer = optim.SGD(output.parameters(), lr=base_lr,
    #                       momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    ce_loss = CrossEntropyLoss()
    bce_loss = BCEWithLogitsLoss()
    dice_loss = losses.DiceLoss(num_classes)

    logger.info("post process: {} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    # best_performance = 0.0

    a_dice = args.loss_weight[1]
    a_bce = args.loss_weight[0]
    max_epoch = args.epoch_num
    total_iterations = max_epoch * len(trainloader)
    print(total_iterations)

    best_model_state = copy.deepcopy(model.state_dict())
    model.train()
    # iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in range(max_epoch):

        tbar = tqdm(trainloader, desc='epoch {}'.format(epoch_num))
        loss_list = []
        ce_loss_list = []
        dice_loss_list = []
        loss_old = []
        aux_iter = iter(trainloader_origin)

        for i_batch, sampled_batch in enumerate(tbar):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label'].squeeze(1)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda().to(torch.float32)
            mask = sampled_batch['mask'].cuda()

            try:
                sampled_batch_ori = next(aux_iter)
            except StopIteration:
                aux_iter = iter(trainloader_origin)
                sampled_batch_ori = next(aux_iter)

            volume_batch_ori, label_batch_ori = sampled_batch_ori['image'], sampled_batch_ori['label'].squeeze(1)
            volume_batch_ori, label_batch_ori = volume_batch_ori.cuda(), label_batch_ori.cuda()


            outputs = model(volume_batch)
            outputs_ori = model(volume_batch_ori)

            if isinstance(outputs, tuple):
                outputs = outputs[0]
                outputs_ori = outputs_ori[0]

            loss_bce = losses.masked_bce_loss(outputs, label_batch.unsqueeze(1), mask.unsqueeze(1))
            loss_bce_1 = bce_loss(outputs, label_batch.unsqueeze(1))
            loss_bce_ori = bce_loss(outputs_ori, label_batch_ori.unsqueeze(1))

            outputs_soft = torch.sigmoid(outputs)
            outputs_ori_soft = torch.sigmoid(outputs_ori)

            loss_dice = losses.masked_dice_loss(
                outputs_soft, label_batch.unsqueeze(1), mask.unsqueeze(1))
            loss_dice_1 = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss_dice_ori = dice_loss(outputs_ori_soft, label_batch_ori.unsqueeze(1))

            supervised_loss_ori = a_dice * loss_dice_ori + a_bce * loss_bce_ori

            dice_loss_list.append(loss_dice.item())
            ce_loss_list.append(loss_bce.item())

            supervised_loss_1 = a_dice * loss_dice + a_bce * loss_bce
            supervised_loss_old = a_dice * loss_dice_1 + a_bce * loss_bce_1

            supervised_loss = 0.7 * supervised_loss_ori + 0.3 * supervised_loss_1

            loss = supervised_loss
            loss_list.append(loss.item())
            # loss_old.append(supervised_loss_old.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / total_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

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

        model.eval()
        metric_list = []
        for i_batch, sampled_batch in enumerate(valloader):
            metric_i, _, _ = test_single_volume(sampled_batch, model)
            metric_list.append( (metric_i) )

        metric_list = np.array(metric_list)

        # dc, mIoU, p, r, f1

        val_dice = np.mean(metric_list, axis=0)[0]
        val_mIoU = np.mean(metric_list, axis=0)[1]

        if val_mIoU > best_performance:
            best_performance = val_mIoU
            best_model_state = copy.deepcopy(model.state_dict())
            # print (performance, iter_num)

            # if val_mIoU > best_performance:
            #     best_performance = val_mIoU
            #     save_mode_path = os.path.join(snapshot_path,
            #                                   'best_ep_{}_iter_{}_mIoU_{}.pth'.format(
            #                                       epoch_num, iter_num, round(best_performance, 4)))
            #     save_best = os.path.join(snapshot_path,
            #                              '{}_best_model.pth'.format(args.model))
            #     torch.save(model.state_dict(), save_mode_path)
            #     torch.save(model.state_dict(), save_best)
            #     logger.info("best found at epoch {}".format(epoch_num))

        logger.info(
            'epoch %d, iteration %d, loss : %f, mean_dice : %f, mean_mIoU : %f' % (epoch_num, iter_num, np.mean(loss_list), val_dice, val_mIoU))
        # print ('epoch %d, iteration %d, loss : %f, mean_dice : %f, mean_mIoU : %f' % (epoch_num, iter_num, np.mean(loss_list), val_dice, val_mIoU))
        model.train()

        if iter_num >= max_iterations:
            break

        # logging.warning(' ep %d : loss : %f, loss_ce: %f, loss_dice: %f' %(epoch_num, tot_loss/len(trainloader), tot_ce_loss/len(trainloader), tot_dice_loss/len(trainloader)))

        # if epoch_num !=0 and epoch_num % 10 == 0:
        #     save_mode_path = os.path.join(
        #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logger.info("save output to {}".format(save_mode_path))

    # print (iter_num)
    return best_model_state, best_performance

if __name__ == '__main__':
    # FLAGS = parser.parse_args()

    snapshot_path = "../output/{}/{}".format(args.exp, args.model)
    os.makedirs(os.path.join(snapshot_path, 'post'), exist_ok=True)
    log_path = os.path.join(snapshot_path, 'post')
    saved_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))

    total_iternum = 60
    high = 0.8
    low = 0.1
    high_step = 0.01

    # logger settings
    global logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(log_path ,'{}_best_model_final_{}_{}iters,{},{},0.7:0.3.txt')\
                                       .format(args.model,args.dataset,total_iternum,high, high_step))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # weight loading
    net = net_factory(net_type=args.model, in_chns=3,
                      class_num=args.num_classes)
    net.load_state_dict(torch.load(saved_mode_path))
    print("init weight from {}".format(saved_mode_path))


    save_mode_path = os.path.join(
        snapshot_path, 'post', '{}_best_model_final_{}_{}iters,{},{}.pth'.format(args.model,args.dataset,total_iternum,high, high_step))
    save_path = os.path.join("../output/{}/{}_predictions".format(args.exp, args.model), '{}_{}iters,{},{},0.7:0.3'\
                                       .format(args.dataset, total_iternum,high, high_step) )

    net_dict = {}
    best_miou_per_iter = 0.0

    for iternum in range(0, total_iternum+1):
        # threshold_post = high - iternum * ( high - low ) / (total_iternum)
        mask_range = [low, high]
        metric, new_dataset = Inference(args, snapshot_path, net, iternum, mask_range)
        net_dict, best_miou_per_iter = post_train(args, snapshot_path, new_dataset, net, best_performance=best_miou_per_iter)

        net.load_state_dict(net_dict)
        # low = low + 0.04
        if iternum % (total_iternum//10) == 0:
            high = high - high_step
            args.base_lr = args.base_lr * 0.8

    torch.save(net_dict, save_mode_path)
    metric, new_dataset = Inference(args, save_path, net, total_iternum, [0,0], need_save=True)




