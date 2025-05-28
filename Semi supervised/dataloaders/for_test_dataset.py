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

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        # img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        blur = transforms.GaussianBlur(kernel_size=5, sigma=sigma)
        img = blur(img)
    return img


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        # img = np.array(img)
        # mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            # value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
            value = torch.empty(erase_h, erase_w, img_c).uniform_(value_min, value_max)
        else:
            # value = np.random.uniform(value_min, value_max)
            value = torch.empty(1).uniform_(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 1

        # img = Image.fromarray(img.astype(np.uint8))
        # mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask

def StrongAugment(sample):
    image, label = sample["image"], sample["label"]

    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    if random.random() < 0.8:
        image = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image)
    image = transforms.RandomGrayscale(p=0.2)(image)
    image = blur(image, p=0.5)
    image, label = cutout(image, label, p=0.5)
    image = image.type(torch.float32)
    label = label.type(torch.uint8)
    sample = {"image": image, "label": label}
    return sample

def preprocess(img_dir):
    img_files = [file for file in os.listdir(img_dir) if file.endswith('.jpg')]

    return sorted(img_files)


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

        self.img_data_dir = self._base_dir
        self.img_path_list = preprocess(self.img_data_dir)

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
        mask_path = self.img_path_list[idx]

        image = Image.open(os.path.join(self.img_data_dir, image_path))

        sample = {}

        # if image.size()[0] > image.size()[1]:
        #     image = torch.transpose(image, 1, 2)
        #     mask = torch.transpose(mask, 1, 2)

        # image1 = np.array(image)
        # mask1 = np.array(mask)
        image = self.transform(image)
        if image.shape[0] == 1:
            image = torch.cat([image] * 3, dim=0)

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
        # sample["label"] = mask
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








