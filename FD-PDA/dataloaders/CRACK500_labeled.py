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

def preprocess(dir):
    jpg_files = [file for file in os.listdir(dir) if file.endswith('.jpg')]
    png_files = [file for file in os.listdir(dir) if file.endswith('.png')]
    return sorted(jpg_files), sorted(png_files)



class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.split = split
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        # self.reliable_path = os.path.join(self._base_dir, "reliable_name.json")
        self.pseudo_labeled_name = []



        self.data_dir = os.path.join(self._base_dir, self.split+'crop')
        self.img_path_list, self.mask_path_list = preprocess(self.data_dir)

        if transform == 'weak' and self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomApply( transforms.RandomRotation([0,90,180,270] ), p=0.5),
                transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
                transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])




    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):

        image_path = self.img_path_list[idx]
        mask_path = self.mask_path_list[idx]

        image = Image.open(os.path.join(self.data_dir, image_path))
        mask = Image.open(os.path.join(self.data_dir, mask_path)).convert('L')

        sample = {}


        # image1 = np.array(image)
        # mask1 = np.array(mask)
        image, mask = self.transform(image, mask)


        # image1 = transforms.ToPILImage()(image)
        # mask1 = transforms.ToPILImage()(mask)
        # image1.save(os.path.join(out_path, image_path))
        # mask1.save(os.path.join(out_path, mask_path))
        # print(2)

        # fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        # axes[0][0].imshow(image1)
        # axes[0][0].axis('off')
        # axes[0][1].imshow(image.permute((1, 2, 0)))
        # axes[0][1].axis('off')
        # axes[1][0].imshow(mask1)
        # axes[1][0].axis('off')
        # axes[1][1].imshow(mask.permute((1, 2, 0)))
        # axes[1][1].axis('off')
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


if __name__ == '__main__':

    pass


