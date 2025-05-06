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
import augmentations
from augmentations.ctaugment import OPS
import json
import matplotlib.pyplot as plt
from PIL import Image
import math

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
        self.reliable_path = os.path.join(self._base_dir, "reliable_name.json")


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
        with open(self.reliable_path, 'r') as fr:
            reliable_json = json.load(fr)


        if self.split == "train": # 1
            self.split_idx = self.split_json['train_unlabeled_idx']
            self.sample_name = [self.sample_name[i] for i in self.split_idx]

        if self.split == "val" or self.split == 'test':
            pass

        if self.split == "label_all" or self.split =='label_semi':
            if addition == 'reliable': # 3
                reliable_name = reliable_json['reliable_name']
                self.sample_name = reliable_name

            if addition == 'unreliable': # 5
                reliable_name = reliable_json['unreliable_name']
                self.sample_name = reliable_name
            if addition == 'all': # 2
                self.split_idx = self.split_json['train_unlabeled_idx']
                self.sample_name = [self.sample_name[i] for i in self.split_idx]

        if self.split == 'retrain': # 4 6
            self.split_idx = self.split_json['train_unlabeled_idx']
            self.sample_name = [self.sample_name[i] for i in self.split_idx]




    def __len__(self):
        return len(self.sample_name)

    def __getitem__(self, idx):

        case = self.sample_name[idx]

        # image_npz = np.load(os.path.join(self._base_dir, "images", case + ".npz"))
        # label_npz = np.load(os.path.join(self._base_dir, "annotations", case + ".npz"))
        # image = image_npz["data"]
        # label = label_npz["data"]

        image = np.load(os.path.join(self._base_dir, "images", case + ".npy"))
        label = np.load(os.path.join(self._base_dir, "annotations", case + ".npy"))

        sample = {"image": image, "label": label}
        if self.split == "train" or self.split =='retrain' :
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)


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


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


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


# class TwoStreamBatchSampler(Sampler):
#     """Iterate two sets of indices
#
#     An 'epoch' is one iteration through the primary indices.
#     During the epoch, the secondary indices are iterated through
#     as many times as needed.
#     """
#
#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         self.primary_indices = primary_indices
#         self.secondary_indices = secondary_indices
#         self.secondary_batch_size = secondary_batch_size
#         self.primary_batch_size = batch_size - secondary_batch_size
#
#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0
#
#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = iterate_eternally(self.secondary_indices)
#
#         return (
#             primary_batch + secondary_batch
#             for (primary_batch, secondary_batch) in zip(
#                 grouper(primary_iter, self.primary_batch_size),
#                 grouper(secondary_iter, self.secondary_batch_size),
#             )
#         )
#
#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size


# def iterate_once(iterable):
#     return np.random.permutation(iterable)
#
#
# def iterate_eternally(indices):
#     def infinite_shuffles():
#         while True:
#             yield np.random.permutation(indices)
#
#     return itertools.chain.from_iterable(infinite_shuffles())


# def grouper(iterable, n):
#     "Collect data into fixed-length chunks or blocks"
#     # grouper('ABCDEFG', 3) --> ABC DEF"
#     args = [iter(iterable)] * n
#     return zip(*args)



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



    # trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
    #                          num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    # trainloader_1 = DataLoader(db_train, batch_sampler=None, batch_size=args.labeled_bs,
    #                          num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    #
    # trainloader_labeled = DataLoader(db_train_labeled, batch_sampler=None, batch_size=args.labeled_bs,
    #                                  shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    #
    # trainloader_unlabeled = DataLoader(db_train_unlabeled, batch_sampler=None, batch_size=args.batch_size-args.labeled_bs,
    #                                  shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    #
    # valloader = DataLoader(db_val, batch_size=1, shuffle=False,
    #                        num_workers=0)
    #
    # labelloader_1 = DataLoader(db_label_all, batch_size=1, shuffle=False, num_workers=0)
    # labelloader_2 = DataLoader(db_label_semi_r, batch_size=1, shuffle=False, num_workers=0)
    #
    # retrainloader = DataLoader(db_retrain_all, batch_size=1, shuffle=False)
    #
    # dataloader = trainloader_unlabeled
    #
    # print (len(dataloader))
    # for i_batch, sampled_batch in enumerate(dataloader):
    #
    #     volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    #     volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
    #     unlabeled_volume_batch = volume_batch[args.labeled_bs:]
    #     print(sampled_batch["name"][0])


    # for i_batch, sampled_batch in enumerate(trainloader):
    #     # print(i_batch)
    #
    #     volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    #     volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
    #     unlabeled_volume_batch = volume_batch[args.labeled_bs:]
    #
    #     noise = torch.clamp(torch.randn_like(
    #         unlabeled_volume_batch) * 0.1, -0.2, 0.2)
    #     ema_inputs = unlabeled_volume_batch + noise
    # args = parse_args()
    #
    # if not os.path.exists(args.input_dir):
    #     raise ValueError('Input directory does not exist.')
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    #
    # preprocess_data(input_path=args.input_dir,
    #                 output_path=args.output_dir,
    #                 split_file=args.split_file)
