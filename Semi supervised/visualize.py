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
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.optim import lr_scheduler

from networks.net_factory import net_factory
from utils import losses, metrics, ramps

import seaborn as sns
import os
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.transform import resize
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from configs.config_supervised_for_debug import args
# from configs.config_supervised_SCSegamba_for_Deepcrack_test import args
# from configs.config_supervised_deepcrack_test import args

# for debug
# from configs.config_supervised_for_debug import args


from dataloaders.CFD_labeled import BaseDataSets as CFD
from dataloaders.DeepCrack_labeled import BaseDataSets as DeepCrack
from dataloaders.CRACK500_labeled import BaseDataSets as CRACK500


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def extract_glcm_features(pil_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    从 PIL 图像提取 GLCM 特征。
    Args:
        pil_image (PIL.Image): 输入的 PIL 图像。
        distances (list): 计算 GLCM 的距离列表。
        angles (list): 计算 GLCM 的角度列表。
    Returns:
        np.ndarray: 提取的 GLCM 特征向量。如果无法处理图像，返回 None。
    """
    try:
        # 如果图像不是灰度图，则转换为灰度图
        if pil_image.mode != 'L':
            img_gray = pil_image.convert('L')
        else:
            img_gray = pil_image

        # 统一缩放图像尺寸以加速 GLCM 计算并保持一致性
        img_resized = np.array(img_gray.resize((128, 128)))
        # img_resized = img_gray

        # 将灰度级量化到较少级别 (例如 64 级)，以减少 GLCM 矩阵大小
        bins = np.linspace(0, 256, 65, endpoint=True, dtype=np.uint8) # 64 bins
        img_quantized = np.digitize(img_resized, bins) - 1
        print (9999)
        img_quantized[img_quantized < 0] = 0
        img_quantized[img_quantized >= len(bins)-1] = len(bins)-2 # 确保最大值在范围内

        # 计算 GLCM 矩阵
        glcm = graycomatrix(img_quantized, distances=distances, angles=angles,
                            levels=len(bins)-1, symmetric=True, normed=True)

        # 提取 GLCM 属性
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        feature_vector = []
        for prop in properties:
            feature_vector.extend(graycoprops(glcm, prop).flatten())

        return np.array(feature_vector)
    except Exception as e:
        # print(f"无法提取 GLCM 特征: {e}") # 生产环境中可以取消注释以调试
        return None

def extract_hog_features(pil_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    从 PIL 图像提取 HOG 特征。
    Args:
        pil_image (PIL.Image): 输入的 PIL 图像。
        pixels_per_cell (tuple): 每个单元格的像素数。
        cells_per_block (tuple): 每个块的单元格数。
        orientations (int): 梯度方向的数量。
    Returns:
        np.ndarray: 提取的 HOG 特征向量。如果无法处理图像，返回 None。
    """
    try:
        # 如果图像不是灰度图，则转换为灰度图
        if pil_image.mode != 'L':
            img_gray = pil_image.convert('L')
        else:
            img_gray = pil_image

        # 统一缩放图像尺寸以保持 HOG 计算的一致性
        img_resized = np.array(img_gray.resize((256, 256)))

        # 计算 HOG 特征
        hog_features = hog(img_resized, orientations=orientations,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           transform_sqrt=True, block_norm='L2-Hys',
                           feature_vector=True)
        return hog_features
    except Exception as e:
        # print(f"无法提取 HOG 特征: {e}") # 生产环境中可以取消注释以调试
        return None

def tensor_to_pil(img_tensor):
    """
    将 PyTorch Tensor (C, H, W) 转换为 PIL Image (RGB 或 L)。
    假定 Tensor 值已归一化 (例如，0-1 或 ImageNet 归一化)。
    如果使用了 ImageNet 归一化，将进行反归一化。
    """
    # 假设您的 DataLoader 使用了 ImageNet 归一化。
    # 如果您的 DataLoader 使用了不同的归一化方式，请修改或移除以下均值和标准差。
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 检查是 3 通道 (RGB) 还是 1 通道 (灰度) 图像
    if img_tensor.shape[0] == 3: # RGB
        # 反归一化并从 [0,1] 范围转换为 [0,255]
        img_tensor = img_tensor * std + mean
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy() # C,H,W 转换为 H,W,C
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np, 'RGB')
    elif img_tensor.shape[0] == 1: # 灰度
        img_np = img_tensor.squeeze(0).cpu().numpy() # 移除通道维度
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np, 'L')
    else:
        raise ValueError(f"不支持的 Tensor 形状转换: {img_tensor.shape}。期望 (1, H, W) 或 (3, H, W)。")


def extract_features_from_dataloader(dataloader, feature_extractor_glcm, feature_extractor_hog):
    glcm_feats = []
    hog_feats = []
    print(f"开始从 DataLoader 提取特征...")
    for batch_idx, sampled_batch in enumerate(dataloader):
        images = sampled_batch['image']
        print(f"  处理批次 {batch_idx+1}/{len(dataloader)}")
        for img_tensor in images:
            # 将 Tensor 图像转换为 PIL Image 以便传统特征提取函数使用
            pil_img = tensor_to_pil(img_tensor)

            glcm_f = feature_extractor_glcm(pil_img)
            hog_f = feature_extractor_hog(pil_img)

            if glcm_f is not None:
                glcm_feats.append(glcm_f)
            if hog_f is not None:
                hog_feats.append(hog_f)
    return np.array(glcm_feats), np.array(hog_feats)

def haar_wavelet_transform(image):
    """
    对输入的PIL图像进行Haar小波变换，返回低频和高频分量。
    """
    import pywt
    import numpy as np
    # 转为灰度
    image_gray = image.convert('L')
    img_arr = np.array(image_gray)
    # 进行Haar小波分解
    coeffs2 = pywt.dwt2(img_arr, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

if __name__ == "__main__":

    cfd_path = os.path.join('../dataset', 'CFD')
    deepcrack_path = os.path.join('../dataset', 'DeepCrack')
    crack500_path = os.path.join('../dataset', 'CRACK500')
    cfd = CFD(base_dir=cfd_path, split="train", transform=None)
    deepcrack_train = DeepCrack(base_dir=deepcrack_path, split="train", transform=None)
    deepcrack_test = DeepCrack(base_dir=deepcrack_path, split="test", transform=None)
    deepcrack = ConcatDataset([deepcrack_train, deepcrack_test])
    crack500_train = CRACK500(base_dir=crack500_path, split="train", transform=None)
    crack500_test = CRACK500(base_dir=crack500_path, split="test", transform=None)
    crack500 = ConcatDataset([crack500_train, crack500_test])

    cfd_loader = DataLoader(cfd, batch_size = 1, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=False)
    deepcrack_loader = DataLoader(deepcrack, batch_size = 1, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)
    crack500_loader = DataLoader(crack500, batch_size = 1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)


    # glcm_features_cfd, hog_features_cfd = extract_features_from_dataloader(cfd_loader, extract_glcm_features, extract_hog_features)
    # glcm_features_deepcrack, hog_features_deepcrack = extract_features_from_dataloader(deepcrack_loader, extract_glcm_features, extract_hog_features)
    # glcm_features_crack500, hog_features_crack500 = extract_features_from_dataloader(crack500_loader, extract_glcm_features, extract_hog_features)
    #
    #
    # combined_features_A = hog_features_cfd
    # combined_features_B = hog_features_deepcrack
    # combined_features_C = hog_features_crack500
    #
    # # combined_features_A = np.hstack((glcm_features_cfd, hog_features_cfd))
    # # combined_features_B = np.hstack((glcm_features_deepcrack, hog_features_deepcrack))
    # # combined_features_C = np.hstack((glcm_features_crack500, hog_features_crack500))
    #
    # all_combined_features = np.vstack((combined_features_A, combined_features_B, combined_features_C))
    # labels = np.array(['CFD'] * len(combined_features_A) +
    #                   ['Deepcrack'] * len(combined_features_B) +
    #                   ['Crack500'] * len(combined_features_C))
    #
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(all_combined_features)
    #
    # tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=1000)
    # tsne_results = tsne.fit_transform(scaled_features)
    # fig = plt.figure(figsize=(10, 8))
    # # 'hue' 参数会自动处理多个类别
    # sns.scatterplot(
    #     x=tsne_results[:, 0], y=tsne_results[:, 1],
    #     hue=labels,
    #     palette=sns.color_palette("hls", 3), # 调整调色板大小以适应 3 个数据集
    #     legend="full",
    #     alpha=0.7
    # )
    # # plt.xlabel('t-SNE 1')
    # # plt.ylabel('t-SNE 2')
    # plt.legend(fontsize=15)
    #
    # ax = plt.gca()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.savefig('tsne_features.png', bbox_inches='tight', pad_inches=0.1)
    # plt.show()



    # db_val = BaseDataSets(base_dir=args.data_path, split="val", transform=None)

    # 新增：对CFD数据集的图片进行haar小波变换并保存分量
    import torch
    from networks.wavelet_diy import HaarWaveletTransform
    haar_save_dir = './haar_wavelet_deepcrack'
    os.makedirs(haar_save_dir, exist_ok=True)
    haar_transform = HaarWaveletTransform()
    for idx, data in enumerate(deepcrack_loader):
        # img为dict或tuple，通常包含'image'和'label'等
        img_tensor = data['image']
        name = data['name'][0]


        # 兼容DataLoader输出为batch的情况
        if hasattr(img_tensor, 'shape'):
            if len(img_tensor.shape) == 4:
                # (B, C, H, W)，取第一个
                img_tensor = img_tensor[0]
        # 现在img_tensor应为(C, H, W)
        img_tensor = img_tensor.float()
        # 保证高宽为偶数
        c, h, w = img_tensor.shape
        if h % 2 != 0 or w % 2 != 0:
            img_tensor = img_tensor[:, :h//2*2, :w//2*2]
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        low_freq, high_freq, freq_list = haar_transform(img_tensor)
        # 保存分量为可视化图像
        def save_img(arr, path):
            arr = arr.squeeze()
            arr = arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
            arr = np.transpose(arr, (1, 2, 0))
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            arr = arr.astype(np.uint8)
            plt.imsave(path, arr, cmap='gray')
        save_img(freq_list[0], os.path.join(haar_save_dir, f'{name}_L.png'))
        save_img(freq_list[1], os.path.join(haar_save_dir, f'{name}_LH.png'))
        save_img(freq_list[2], os.path.join(haar_save_dir, f'{name}_HL.png'))
        save_img(freq_list[3], os.path.join(haar_save_dir, f'{name}_HH.png'))
        save_img(high_freq, os.path.join(haar_save_dir, f'{name}_H.png'))
