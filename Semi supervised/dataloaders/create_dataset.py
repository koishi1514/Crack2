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


class NewDataSets(Dataset):
    def __init__(
            self,
            img_list,
            label_list,
            name_list,
            mask_list=None,
            real_label_list=None,
            base_dir=None,
            split="train",
            transform=None,
    ):
        self._base_dir = base_dir
        self.split = split


        self.img_list = img_list
        self.label_list = label_list
        self.name_list = name_list
        self.mask_list = mask_list
        self.real_label_list = real_label_list

        if transform == 'weak' and 0: # never aug
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
        return len(self.img_list)

    def __getitem__(self, idx):

        image = self.img_list[idx]
        mask = self.label_list[idx]
        name = self.name_list[idx]
        mask_post = self.mask_list[idx]
        real_label = self.real_label_list[idx]

        # image: c, h ,w -> h, w, c
        image = image.transpose(1, 2, 0)
        sample = {}

        # if image.size()[0] > image.size()[1]:
        #     image = torch.transpose(image, 1, 2)
        #     mask = torch.transpose(mask, 1, 2)

        # image1 = np.array(image)
        # mask1 = np.array(mask)
        image, mask = self.transform(image, mask)

        if image.shape[0] == 1:
            image = torch.cat([image] * 3, dim=0)

        mask1 = (mask > 0.5).float()
        image1 = transforms.ToPILImage()(image)
        mask1 = transforms.ToPILImage()(mask1)

        mask_post1 = mask_post
        # image1.save(os.path.join(out_path, image_path))
        # mask1.save(os.path.join(out_path, mask_path))
        # print(2)

        # fig, axes = plt.subplots(1, 4, figsize=(12, 6))
        # axes[0].imshow(image1)
        # axes[0].axis('off')
        # axes[1].imshow(mask1, cmap='gray')
        # axes[1].axis('off')
        # axes[2].imshow(mask_post1, cmap='gray')
        # axes[2].axis('off')
        # axes[3].imshow(real_label, cmap='gray')
        # axes[3].axis('off')
        #
        # plt.tight_layout()
        # plt.show()
        # plt.close()

        # if self.split == 'retrain' and case in self.pseudo_labeled_name:
        #     sample = StrongAugment(sample)

        # sample = StrongAugment(sample)

        sample["image"] = image
        sample["label"] = mask
        sample["idx"] = idx
        sample["name"] = name
        sample["mask"] = mask_post
        return sample









