import os

import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset

import torchvision.transforms.v2 as transforms
from scipy import ndimage
from torch.utils.data.sampler import Sampler
# import augmentations
# from augmentations.ctaugment import OPS
import json
import matplotlib.pyplot as plt
from PIL import Image
import math
from dataloaders.dataloader_utils import get_img_filenames


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        ops_weak=None,
        ops_strong=None,
        addition = None,
        labeled_num = 400
    ):
        self._base_dir = base_dir
        self.split = split
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        # self.reliable_path = os.path.join(self._base_dir, "reliable_name.json")
        self.pseudo_labeled_name = []


        # self.name_json_path = os.path.join(self._base_dir, "name.json")
        # self.split_json_path = os.path.join(self._base_dir, "frame_split_{}.json".format(labeled_num) )

        # with open(self.name_json_path, 'r') as f:
        #     name_json = json.load(f)
        # self.sample_name = name_json['file_name']
        # self.sample_idx = self.sample_idx_labeled + self.sample_idx_unlabeled


        # with open(self.split_json_path, 'r') as f1:
        #     self.split_json = json.load(f1) # number idx
        # self.img_data_dir = os.path.join(self._base_dir, "train"+"_img")
        # self.label_dir = os.path.join(self._base_dir, "train"+"_lab")
        self.img_data_dir = os.path.join(self._base_dir, "images")
        self.label_dir = os.path.join(self._base_dir, "binary_labels")
        self.label_dir_1 = os.path.join(self._base_dir, "labels")

        self.img_path_list = get_img_filenames(self.img_data_dir, '.jpg')
        self.mask_path_list = get_img_filenames(self.label_dir, '.jpg')
        self.mask_path_list_1 = get_img_filenames(self.label_dir_1, '.jpg')


        if transform == 'weak':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomApply(transforms.RandomRotation([0,90,180,270]), p=0.5),
                transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
                transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])


        # if self.split == "train":
        #     self.split_idx = self.split_json['train_labeled_idx']
        #     self.sample_name = [self.sample_name[i] for i in self.split_idx]
        #
        # if self.split == "val":
        #     self.split_idx = self.split_json['{}_idx'.format(self.split)]
        #     self.sample_name = [self.sample_name[i] for i in self.split_idx]
        #
        # if self.split == 'test':

        # if self.split == "label_all" or self.split =='label_semi':
        #     pass

        # if self.split == 'retrain':
        #     if addition == 'all':
        #         # labeled_idx = self.split_json['train_labeled_idx']
        #         unlabeled_idx = self.split_json['train_unlabeled_idx']
        #         self.split_idx = self.split_json['train_labeled_idx'] + self.split_json['train_unlabeled_idx']
        #         self.pseudo_labeled_name = [self.sample_name[i] for i in unlabeled_idx]
        #         self.sample_name = [self.sample_name[i] for i in self.split_idx]
        #
        #     elif addition == 'reliable':
        #
        #         with open(self.reliable_path, 'r') as fr:
        #             reliable_json = json.load(fr)
        #
        #         # labeled_idx = self.split_json['train_labeled_idx']
        #
        #         reliable_name = reliable_json['reliable_name']
        #         self.split_idx = self.split_json['train_labeled_idx']
        #         self.pseudo_labeled_name = reliable_name
        #         self.sample_name = [self.sample_name[i] for i in self.split_idx] + reliable_name



    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):

        image_path = self.img_path_list[idx]
        mask_path = self.mask_path_list[idx]
        # mask1_path = self.mask_path_list_1[idx]

        image = Image.open(os.path.join(self.img_data_dir, image_path))
        mask = Image.open(os.path.join(self.label_dir, mask_path))
        # mask1 = Image.open(os.path.join(self.label_dir_1, mask1_path))

        mask_np = np.array(mask) # mask_np 的形状会是 (height, width, 3)

        # 3. 分离 RGB 通道
        R = mask_np[:, :, 0] # 红色通道
        G = mask_np[:, :, 1] # 绿色通道
        B = mask_np[:, :, 2] # 蓝色通道
        red_threshold = 150
        color_dominance_margin = 30

        is_red_pixel = (R > red_threshold) & \
                       (R > G + color_dominance_margin) & \
                       (R > B + color_dominance_margin)
        extracted_red_mask_np = np.zeros_like(R, dtype=np.uint8)

        #    将符合“红色”条件的像素设置为白色 (255)
        extracted_red_mask_np[is_red_pixel] = 255

        # 6. 将 NumPy 数组转换回 PIL 图像的 'L' 模式 (灰度图)
        mask = Image.fromarray(extracted_red_mask_np, 'L')

        sample = {}


        # if image.size()[0] > image.size()[1]:
        #     image = torch.transpose(image, 1, 2)
        #     mask = torch.transpose(mask, 1, 2)

        # image1 = np.array(image)
        # mask1 = np.array(mask)
        image, mask = self.transform(image, mask)
        # mask1 = self.transform(mask1)


        # image1 = transforms.ToPILImage()(image)
        # mask1 = transforms.ToPILImage()(mask)
        # image1.save(os.path.join(out_path, image_path))
        # mask1.save(os.path.join(out_path, mask_path))
        # print(2)

        # fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        #
        # axes[0][0].imshow(image.permute((1, 2, 0)))
        # axes[0][0].axis('off')
        # axes[1][0].imshow(mask1.permute((1, 2, 0)))
        # axes[1][0].axis('off')
        # axes[0][1].imshow(mask.permute((1, 2, 0)), cmap ='grey')
        # axes[0][1].axis('off')
        # plt.tight_layout()
        # plt.show()
        # plt.close()

        # if self.split == 'retrain' and case in self.pseudo_labeled_name:
        #     sample = StrongAugment(sample)

        # sample = StrongAugment(sample)

        sample["image"] = image
        sample["label"] = mask
        sample["idx"] = idx
        sample["name"] = image_path
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)




def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

if __name__ == '__main__':

    from configs.config_supervised_real_data_test import args
    from torch.utils.data import DataLoader

    data_path = os.path.join('..',args.data_path)
    out_path = os.path.join('..', 'testout')
    batch_size = 10

    # split_json_path = os.path.join(root_path, "frame_split.json")
    # with open(split_json_path, 'r') as f:
    #     split_dict = json.load(f)


    db_train = BaseDataSets(base_dir=data_path, split="train", transform=None)
    trainloader = DataLoader(db_train, batch_size = batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=False)

    for i_batch, sampled_batch in enumerate(trainloader):
        # pass
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        # print(volume_batch.shape,label_batch.shape)

        # print(i_batch, sampled_batch['name'])


    # db_val = BaseDataSets(base_dir=root_path, split="val")
    # db_test = BaseDataSets(base_dir=root_path, split="test")
    # db_retrain_st = BaseDataSets(base_dir=root_path, split="retrain", num=None, transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)]), addition = 'all')
    #
    # db_retrain_st_plus = BaseDataSets(base_dir=root_path, split="retrain", num=None, transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)]), addition = 'reliable')






