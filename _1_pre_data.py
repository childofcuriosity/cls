import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io as sio
import os
from scipy.io import loadmat
from PIL import Image
import random
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, samples, transforms=None): # samples是list，元素是(img,label)
        self.transforms = transforms
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img,label = self.samples[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label



test_labels=loadmat('data/test_annot.mat')['annot2'].astype(np.float32)
with open('data/test_list.txt', 'r') as f:
    test_data_paths = ['data/'+line[:-1]+'.jpeg' for line in f.readlines()]

test_datas = []
for img_path in test_data_paths:
    img = Image.open(img_path).convert('RGB')
    if img.size == (128, 192):  # 判断图像尺寸是否为 (128, 192)
        img = np.array(img)
        img = img.transpose((1, 0, 2))  # 转置以得到 (192, 128) 的形状
        img = Image.fromarray(img)
    test_datas.append(img)

test_samples=list(zip(test_datas, test_labels))

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

test_set = CustomDataset(test_samples,test_transforms)

torch.save(test_set,'test_set.pt')


all_train_labels=loadmat('data/train_annot.mat')['annot1'].astype(np.float32)
with open('data/train_list.txt', 'r') as f:
    all_train_data_paths = ['data/'+line[:-1]+'.jpeg' for line in f.readlines()]
all_train_datas = []
for img_path in all_train_data_paths:
    img = Image.open(img_path).convert('RGB')
    if img.size == (128, 192):  # 判断图像尺寸是否为 (128, 192)
        img = np.array(img)
        img = img.transpose((1, 0, 2))  # 转置以得到 (192, 128) 的形状
        img = Image.fromarray(img)
    all_train_datas.append(img)

all_train_samples=list(zip(all_train_datas, all_train_labels))


train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(128, 192), scale=(0.9, 1.1), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10), fill=(0.5, 0.5, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


all_train_set = CustomDataset(all_train_samples,train_transforms)

torch.save(all_train_set,'all_train_set.pt')


random.seed(0)

random.shuffle(all_train_samples)

valid_samples=all_train_samples[:500]
train_samples=all_train_samples[500:]

train_set = CustomDataset(train_samples,train_transforms)
valid_set = CustomDataset(valid_samples,test_transforms)

torch.save(train_set,'train_set.pt')
torch.save(valid_set,'valid_set.pt')





