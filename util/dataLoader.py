# -*- coding: utf-8 -*-
# @Time    : 2022/12/31 12:23
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import platform

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from net_env.net_env import BATCH_SIZE
from util.dataImageAugmentation import train_transforms, val_transforms, test_transforms
from util.dataProcess import PreProcess


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        if platform.system() == 'Linux':
            splitSig = '/'
        else:
            splitSig = "\\"

        label = img_path.split(splitSig)[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


def getDataLoader(mode='train'):
    if mode == 'train':
        train_list, valid_list = PreProcess(mode='train')
        train_data = CatsDogsDataset(train_list, transform=train_transforms)
        valid_data = CatsDogsDataset(valid_list, transform=val_transforms)
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)
        # print(len(train_data), len(train_loader))
        # print(len(valid_data), len(valid_loader))
        return train_loader, valid_loader
    else:
        test_list = PreProcess(mode='test')
        test_data = CatsDogsDataset(test_list, transform=test_transforms)
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
        # print(len(test_data), len(test_loader))
        return test_loader


if __name__ == '__main__':
    _1, _2 = getDataLoader('train')
    _3 = getDataLoader('test')
