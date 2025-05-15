import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

from utils.metrics import calculate_metric_percase_val

def calculate_metric_percase(pred, gt):
    pred[pred > 0.5] = 1
    gt[gt > 0.5] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    # drop batch
    # image = image.squeeze(0)
    # label = label.squeeze(0)

    image = image.cuda()
    label = label.cpu().detach().numpy()

    net.eval()
    with torch.no_grad():
        out = net(image)
        if isinstance(out, tuple):
            out = out[0]

        out_soft = torch.sigmoid(out)

        out = out_soft.cpu().detach().numpy()
        prediction = out

    metric_list = []

    prediction = prediction.squeeze()
    label = label.squeeze()

    metric_list.append(calculate_metric_percase_val(prediction, label))
    # dice, mIoU, precision, recall, f1
    return metric_list


if __name__ == '__main__':

    from configs.config_supervised_test import args
    from torch.utils.data import DataLoader
    from dataloaders.Crack500_labeled import BaseDataSets
    from networks.crackformerII import crackformer

    db_val = BaseDataSets(base_dir=args.data_path, split="val", transform=None)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    num_classes = args.num_classes
    in_chns = 3

    model = crackformer(in_channels=in_chns, final_hidden_dims=64, num_classes=1).cuda()
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        metric_i = test_single_volume(
            sampled_batch["image"].cuda(), sampled_batch["label"].cuda(), model, classes=num_classes)
        metric_list += np.array(metric_i)