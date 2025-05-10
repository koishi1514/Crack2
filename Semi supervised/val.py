import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
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
    prediction = np.zeros_like(label)

    net.eval()
    with torch.no_grad():
        out = net(image)
        if isinstance(out, tuple):
            out = out[0]
        out_soft = torch.softmax(out, dim=1)
        out = torch.argmax(out_soft, dim=1)

        out = out.cpu().detach().numpy()
        prediction = out

    metric_list = []

    prediction = prediction.squeeze()
    label = label.squeeze()

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
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

    model = crackformer(in_channels=in_chns, final_hidden_dims=64, num_classes=2).cuda()
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        metric_i = test_single_volume(
            sampled_batch["image"].cuda(), sampled_batch["label"].cuda(), model, classes=num_classes)
        metric_list += np.array(metric_i)