import os
import torch
import random
import json
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


        self.image_dir = os.path.join(self._base_dir, 'image')
        self.gt_dir = os.path.join(self._base_dir, 'gt')

        # 读取由你提供的脚本生成的 JSON 文件
        splits_file = os.path.join(self._base_dir, 'dataset_split.json')

        with open(splits_file, 'r') as f:
            all_splits = json.load(f)

        self.sample_list = all_splits[self.split]


        if transform == 'weak' and self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
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
        return len(self.sample_list)

    def __getitem__(self, idx):

        full_image_name = self.sample_list[idx]

        # 2. 从完整文件名中分离出基本名，例如 'A'
        base_name = os.path.splitext(full_image_name)[0]

        # 3. 构建图像和掩码的完整路径
        image_path = os.path.join(self.image_dir, full_image_name)
        mask_path = os.path.join(self.gt_dir, base_name + '.bmp')
        # --- 修改结束 ---

        # 使用PIL打开图像和掩码
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 应用变换
        if self.transform:
            image, mask = self.transform(image, mask)

        # 归一化和类型转换

        sample = {
            "image": image,
            "label": mask,
            "idx": idx,
            "name": base_name # 返回基本名，方便调试
        }
        return sample

def worker_init_fn(worker_id):
    random.seed(random.randint(0, 1024) + worker_id)

if __name__ == '__main__':
    # --- 使用示例 ---
    data_path = '/path/to/your/new_dataset' # <--- !!! 修改为你的数据集路径 !!!
    batch_size = 4

    print("加载训练集...")
    db_train = BaseDataSets(base_dir=data_path, split="train", transform="weak")
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    # 遍历一个批次进行测试
    sampled_batch = next(iter(trainloader))
    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

    print(f"图像批次尺寸: {volume_batch.shape}")
    print(f"标签批次尺寸: {label_batch.shape}")
    print(f"样本名称: {sampled_batch['name']}")
