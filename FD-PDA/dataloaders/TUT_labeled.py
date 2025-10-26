import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader

# ... 您其他的辅助函数（如 blur, cutout 等）保持不变 ...

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

        # 根据 split 参数确定图像和标签的目录
        if self.split == 'train':
            self.image_dir = os.path.join(self._base_dir, 'train_img')
            self.label_dir = os.path.join(self._base_dir, 'train_lab')
        elif self.split == 'val':
            self.image_dir = os.path.join(self._base_dir, 'val_img')
            self.label_dir = os.path.join(self._base_dir, 'val_lab')
        elif self.split == 'test':
            self.image_dir = os.path.join(self._base_dir, 'test_img')
            self.label_dir = os.path.join(self._base_dir, 'test_lab')


        # 直接从图像目录获取所有 .jpg 文件名
        self.sample_list = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

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
        # 获取图像文件名
        image_filename = self.sample_list[idx]

        image_path = os.path.join(self.image_dir, image_filename)

        # 推断标签文件名并构建完整路径
        label_filename = image_filename.replace('.jpg', '.png')
        label_path = os.path.join(self.label_dir, label_filename)

        # 加载和处理
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)

        sample = {
            "image": image,
            "label": mask,
            "idx": idx,
            "name": image_filename
        }
        return sample

# --- 使用示例（无输出）---
if __name__ == '__main__':
    # 假设你的数据集目录结构如下:
    # ./my_pre_split_dataset/
    #  ├── train_img/, train_lab/
    #  ├── val_img/, val_lab/
    #  └── test_img/, test_lab/

    DATA_ROOT = './my_pre_split_dataset'
    BATCH_SIZE = 4

    # 创建训练集 DataLoader
    db_train = BaseDataSets(
        base_dir=DATA_ROOT,
        split='train',
        transform='weak'
    )
    train_loader = DataLoader(db_train, batch_size=BATCH_SIZE, shuffle=True)

    # 创建验证集 DataLoader
    db_val = BaseDataSets(
        base_dir=DATA_ROOT,
        split='val',
        transform='none'
    )
    val_loader = DataLoader(db_val, batch_size=BATCH_SIZE, shuffle=False)

    # 可以在这里进行静默的迭代测试，以确保代码能正常运行
    if len(db_train) > 0:
        for _ in train_loader:
            # 迭代一个批次后立即退出，仅用于验证
            break
