import argparse
import logging
import os
import random
import shutil
import sys
import time
import importlib

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

# --- 1. 修改导入以匹配新项目 ---
from configs.config_supervised_for_debug import args

# 使用新项目的动态导入方式来加载Dataset类
import_dataset_name = "dataloaders." + args.dataset + "_labeled"
dataset_py = importlib.import_module(import_dataset_name)
BaseDataSets = getattr(dataset_py, "BaseDataSets")


def inspect(dataloader, batch_name, logger, num_samples=10, save_dir=None):
    """
    遍历数据集，随机挑选样本，并将可视化结果保存为文件。
    此版本不使用tqdm，不带标题，并保存到指定目录。

    Args:
        dataloader (DataLoader): 要检查的数据加载器。
        batch_name (str): 数据集的名称 (e.g., "Training Set")。
        logger: 日志记录器。
        num_samples (int): 要随机挑选和可视化的样本的确切数量。
        save_dir (str, optional): 保存可视化结果的目录。如果为None，则不保存。
    """
    logger.info(f"Generating visualization for {num_samples} samples from {batch_name}...")

    dataset_size = len(dataloader.dataset)
    all_indices = list(range(dataset_size))
    target_indices = set(random.sample(all_indices, num_samples))

    images_to_show = []
    labels_to_show = []
    sample_idx_counter = 0

    if num_samples == 0:
        return

    # 1. 移除tqdm
    for batch in dataloader:
        image_batch, label_batch = batch['image'], batch['label']
        current_batch_size = image_batch.shape[0]

        for i in range(current_batch_size):
            if sample_idx_counter in target_indices:
                image_sample = image_batch[i].cpu()
                label_sample = label_batch[i].cpu()

                if image_sample.shape[0] > 1:
                    image_to_plot = image_sample.permute(1, 2, 0).numpy()
                else:
                    image_to_plot = image_sample.squeeze(0).numpy()

                label_to_plot = label_sample.squeeze(0).numpy()

                images_to_show.append(image_to_plot)
                labels_to_show.append(label_to_plot)

            sample_idx_counter += 1

        if len(images_to_show) == num_samples:
            break

    if not images_to_show:
        return

    num_collected = len(images_to_show)

    cols = int(math.ceil(math.sqrt(num_collected)))
    rows = int(math.ceil(num_collected / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)

    # 2. 移除 fig.suptitle
    # fig.suptitle(f"Random {num_collected} Image Samples from {batch_name}", fontsize=16)

    flat_axes = axes.flatten()

    for i in range(num_collected):
        img = images_to_show[i]
        ax = flat_axes[i]
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)

        # 3. 移除 ax.set_title
        # ax.set_title(f"Sample #{i+1}")
        ax.axis('off')

    for j in range(num_collected, len(flat_axes)):
        flat_axes[j].axis('off')

    plt.tight_layout()

    # 4. 将可视化结果保存到文件
    if save_dir:
        # 根据 batch_name 创建一个唯一的文件名
        filename = f"visualization_{batch_name.replace(' ', '_').lower()}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        logger.info(f"Saved visualization to {filepath}")

    # 关闭图像以释放内存，这在保存文件时尤其重要
    plt.close(fig)

def debug_data_loading(args, snapshot_path):
    """
    主函数，用于加载新项目的数据并进行调试检查。
    """
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # --- 数据加载部分保持不变 ---
    db_train = BaseDataSets(base_dir=args.data_path, split="train", transform=None)

    if args.dataset == 'DeepCrack' or args.dataset == 'AEL':
        db_val = BaseDataSets(base_dir=args.data_path, split="train", transform=None)
    else:
        db_val = BaseDataSets(base_dir=args.data_path, split="val", transform=None)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=False)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    logger.info(f"Found {len(db_train)} training samples, creating {len(trainloader)} batches.")
    logger.info(f"Found {len(db_val)} validation samples, creating {len(valloader)} batches.")

    NUM_SAMPLES_TO_VISUALIZE = 36

    # --- 修改点：将 snapshot_path 作为 save_dir 参数传递 ---

    inspect(trainloader, "Training Set", logger, num_samples=NUM_SAMPLES_TO_VISUALIZE, save_dir=snapshot_path)
    inspect(valloader, "Validation Set", logger, num_samples=NUM_SAMPLES_TO_VISUALIZE, save_dir=snapshot_path)

    return "Debug Finished!"



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

    snapshot_path = "../output/{}/{}".format(args.exp, "dataloader_check")
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if not os.path.exists(os.path.join(snapshot_path, 'log')):
        os.makedirs(os.path.join(snapshot_path, 'log'))

    # 日志记录器设置
    global logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(snapshot_path, 'debug_data_log.txt'), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(str(args))

    debug_data_loading(args, snapshot_path)
