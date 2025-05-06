import os
from cProfile import label

import cv2
import torch
import random
import numpy as np
from glob import glob


from torch.utils.data import Dataset
# import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
# import augmentations
# from augmentations.ctaugment import OPS
import json
import matplotlib.pyplot as plt
from PIL import Image
import math

# 未完成版
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

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        addition = None,
        labeled_num = 400
    ):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        # self.reliable_path = os.path.join(self._base_dir, "reliable_name.json")
        self.pseudo_labeled_name = []


        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        self.name_json_path = os.path.join(self._base_dir, "name.json")
        self.split_json_path = os.path.join(self._base_dir, "frame_split_{}.json".format(labeled_num) )

        with open(self.name_json_path, 'r') as f:
            name_json = json.load(f)
        self.sample_name = name_json['file_name']
        # self.sample_idx = self.sample_idx_labeled + self.sample_idx_unlabeled


        with open(self.split_json_path, 'r') as f1:
            self.split_json = json.load(f1) # number idx


        if self.split == "train":
            self.split_idx = self.split_json['train_labeled_idx']
            self.sample_name = [self.sample_name[i] for i in self.split_idx]

        if self.split == "val" or self.split == 'test':
            self.split_idx = self.split_json['{}_idx'.format(self.split)]
            self.sample_name = [self.sample_name[i] for i in self.split_idx]

        if self.split == "label_all" or self.split =='label_semi':
            pass

        if self.split == 'retrain':
            if addition == 'all':
                # labeled_idx = self.split_json['train_labeled_idx']
                unlabeled_idx = self.split_json['train_unlabeled_idx']
                self.split_idx = self.split_json['train_labeled_idx'] + self.split_json['train_unlabeled_idx']
                self.pseudo_labeled_name = [self.sample_name[i] for i in unlabeled_idx]
                self.sample_name = [self.sample_name[i] for i in self.split_idx]

            elif addition == 'reliable':

                with open(self.reliable_path, 'r') as fr:
                    reliable_json = json.load(fr)

                # labeled_idx = self.split_json['train_labeled_idx']

                reliable_name = reliable_json['reliable_name']
                self.split_idx = self.split_json['train_labeled_idx']
                self.pseudo_labeled_name = reliable_name
                self.sample_name = [self.sample_name[i] for i in self.split_idx] + reliable_name




    def __len__(self):
        return len(self.sample_name)

    def __getitem__(self, idx):

        case = self.sample_name[idx]

        # image_npz = np.load(os.path.join(self._base_dir, "images", case + ".npz"))
        # label_npz = np.load(os.path.join(self._base_dir, "annotations", case + ".npz"))
        # image = image_npz["data"]
        # label = label_npz["data"]

        image = np.load(os.path.join(self._base_dir, "images", case + ".npy"))

        if self.split == 'retrain' and case in self.pseudo_labeled_name:
            label = np.load(os.path.join(self._base_dir, "PseudoMask", case + ".npy"))
        else:
            label = np.load(os.path.join(self._base_dir, "annotations", case + ".npy"))

        sample = {"image": image, "label": label}

        if self.split == "train" or self.split =='retrain' :
            if None not in (self.ops_weak, self.ops_strong):
                # 这个分支无效
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

                # if self.split == 'retrain' and case in self.pseudo_labeled_name:
                #     sample = StrongAugment(sample)

                sample = StrongAugment(sample)


        sample["idx"] = idx
        sample["name"] = self.sample_name[idx]
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



class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)




if __name__ == '__main__':

    from configs.config_for_camus_dataset_test import args
    from torch.utils.data import DataLoader


    batch_size = args.batch_size
    root_path = r'D:\work\SSL4MIS-master\data\CAMUS'


    split_json_path = os.path.join(root_path, "frame_split.json")
    with open(split_json_path, 'r') as f:
        split_dict = json.load(f)


    db_train = BaseDataSets(base_dir=root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=root_path, split="val")
    db_test = BaseDataSets(base_dir=root_path, split="test")
    db_retrain_st = BaseDataSets(base_dir=root_path, split="retrain", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)]), addition = 'all')

    db_retrain_st_plus = BaseDataSets(base_dir=root_path, split="retrain", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)]), addition = 'reliable')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)




