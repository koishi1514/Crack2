import os
import torch
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
        self.transform = transform # 变换逻辑先置空，下面再定义
        self.sample_list = []


        # 处理 'val' 到 'valid' 的映射
        split_prefix = self.split
        if self.split == 'val':
            split_prefix = 'valid'

        list_path = os.path.join(self._base_dir, 'test.txt')

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                relative_path = parts[0] # e.g., 'croppedimg/train_0178_1_1.jpg'
                filename = os.path.basename(relative_path) # e.g., 'train_0178_1_1.jpg'

                # 根据文件名和当前 split 筛选样本
                if filename.startswith(split_prefix + '_'):
                    self.sample_list.append(relative_path)

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
        # --- MODIFICATION POINT 3: 调整路径构建方式 ---
        relative_image_path = self.sample_list[idx]

        # 1. 构建完整的图像路径
        image_path = os.path.join(self._base_dir, relative_image_path)

        relative_mask_path = relative_image_path.replace('croppedimg', 'croppedgt').replace('.jpg', '.png')
        mask_path = os.path.join(self._base_dir, relative_mask_path)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)

        sample = {
            "image": image,
            "label": mask,
            "idx": idx,
            "name": os.path.basename(relative_image_path)
        }
        return sample

# --- 使用示例 ---
if __name__ == '__main__':
    # 假设你的数据集目录结构如下:
    # ./new_txt_dataset/
    #  ├── croppedimg/
    #  │   ├── train_001.jpg
    #  │   └── valid_001.jpg
    #  ├── croppedmask/
    #  │   ├── train_001.bmp
    #  │   └── valid_001.bmp
    #  └── split_file.txt  <-- 你的txt文件

    DATA_ROOT = './new_txt_dataset'  # <--- 修改为你的数据集根目录
    LIST_FILE = os.path.join(DATA_ROOT, 'split_file.txt') # <--- 修改为你的txt文件名
    BATCH_SIZE = 4

    # 创建训练集 DataLoader
    print("加载训练集...")
    db_train = BaseDataSets(
        base_dir=DATA_ROOT,
        list_path=LIST_FILE,
        split='train',
        transform='weak'
    )
    train_loader = DataLoader(db_train, batch_size=BATCH_SIZE, shuffle=True)

    # 创建验证集 DataLoader
    print("\n加载验证集...")
    db_val = BaseDataSets(
        base_dir=DATA_ROOT,
        list_path=LIST_FILE,
        split='val', # 注意这里用 'val'
        transform='none'
    )
    val_loader = DataLoader(db_val, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n训练集样本数: {len(db_train)}")
    print(f"验证集样本数: {len(db_val)}")
