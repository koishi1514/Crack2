import os
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader


class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
    ):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(self._base_dir, 'img')
        self.gt_dir = os.path.join(self._base_dir, 'gt')

        # 1. 获取测试样本列表
        test_file_path = os.path.join(self._base_dir, 'test.txt')
        with open(test_file_path, 'r') as f:
            # 读取每一行，并取第一个路径（因为两列内容相同）
            # 使用 set 以提高后续查找效率
            test_samples_set = {line.strip().split()[0].replace('\\', '/') for line in f if line.strip()}

        # 2. 获取所有样本列表 (扩展名为 .jpg)
        all_samples = []
        if os.path.isdir(self.image_dir):
            for sub_dir in sorted(os.listdir(self.image_dir)):
                sub_dir_path = os.path.join(self.image_dir, sub_dir)
                if os.path.isdir(sub_dir_path):
                    for file_name in sorted(os.listdir(sub_dir_path)):
                        if file_name.endswith('.jpg'):
                            relative_path = os.path.join(sub_dir, file_name).replace('\\', '/')
                            all_samples.append(relative_path)

        # 3. 根据 split 参数确定最终的样本列表
        if self.split == 'train':
            # 训练集 = 所有样本 - 测试样本
            self.sample_list = [p for p in all_samples if p not in test_samples_set]
        elif self.split == 'test':
            self.sample_list = sorted(list(test_samples_set))
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train' or 'test'.")

        # 变换逻辑
        if self.transform == 'weak' and self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
                transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        else: # 测试集或不使用弱增强
            self.transform = transforms.Compose([
                transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # sample_list中的元素形如 "AIGLE_RN/xxx.jpg"
        relative_path = self.sample_list[idx]

        # 构建图像和掩码的完整路径
        image_path = os.path.join(self.image_dir, relative_path)

        # 将图像文件名（.jpg）替换为掩码文件名（.png）
        base_relative_path = os.path.splitext(relative_path)[0]
        # --- 修改点：将掩码文件扩展名从 .bmp 修改为 .png ---
        mask_path = os.path.join(self.gt_dir, base_relative_path + '.png')

        # 使用PIL打开图像和掩码
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 应用变换
        if self.transform:
            image, mask = self.transform(image, mask)

        sample = {
            "image": image,
            "label": mask,
            "idx": idx,
            "name": relative_path
        }
        return sample

def worker_init_fn(worker_id):
    random.seed(random.randint(0, 1024) + worker_id)

if __name__ == '__main__':
    # --- 使用示例 ---
    # !!! 请将此路径修改为你的 'AEL' 数据集所在的根目录 !!!
    pass
