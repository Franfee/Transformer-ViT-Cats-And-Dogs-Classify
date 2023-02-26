# -*- coding: utf-8 -*-
# @Time    : 2022/12/31 12:43
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import os
import glob
import platform
import zipfile

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from net_env.net_env import SEED, BASE_DIR


def ExtractData():
    print("Extracting Data...")
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    with zipfile.ZipFile(os.path.join(BASE_DIR, 'data', 'train.zip')) as train_zip:
        train_zip.extractall(os.path.join(BASE_DIR, 'data'))
    print("Extracting train-data done, ready to extract test-data...")
    with zipfile.ZipFile(os.path.join(BASE_DIR, 'data', 'test.zip')) as test_zip:
        test_zip.extractall(os.path.join(BASE_DIR, 'data'))
    print("Extracting Data Done!")


def PreProcess(mode, RandomPlots=False):
    # =====================Load Data======================
    print("Loading data...")
    if mode == 'train':
        train_dir = os.path.join(BASE_DIR, 'data', 'train')
        train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
        print(f"Read Train Data: {len(train_list)}")

        # 路径名字和平台有差别
        if platform.system() == 'Linux':
            splitSig = '/'
        else:
            splitSig = "\\"
        labels = [path.split(splitSig)[-1].split('.')[0] for path in train_list]

        # =====================Random Plots=================
        if RandomPlots:
            random_idx = np.random.randint(1, len(train_list), size=9)
            fig, axes = plt.subplots(3, 3, figsize=(16, 12))
            plt.subplots_adjust(hspace=0.5)
            for idx, ax in enumerate(axes.ravel()):
                img = Image.open(train_list[random_idx[idx]])
                ax.set_title(str(random_idx[idx]) + ":" + labels[random_idx[idx]])
                ax.imshow(img)
            plt.show()
        # =====================split========================
        train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=SEED)

        print(f"Split Train Data: {len(train_list)}")
        print(f"Split Validation Data: {len(valid_list)}")
        return train_list, valid_list
    else:
        test_dir = os.path.join(BASE_DIR, 'data', 'test')
        test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
        print(f"Read Test Data: {len(test_list)}")
        if RandomPlots:
            random_idx = np.random.randint(1, len(test_list), size=9)
            fig, axes = plt.subplots(3, 3, figsize=(16, 12))
            plt.subplots_adjust(hspace=0.5)
            for idx, ax in enumerate(axes.ravel()):
                img = Image.open(test_list[random_idx[idx]])
                ax.set_title(str(random_idx[idx]))
                ax.imshow(img)
            plt.show()
        return test_list


if __name__ == '__main__':
    # ExtractData()
    _1, _2 = PreProcess(mode='train', RandomPlots=True)
    _3 = PreProcess(mode='test', RandomPlots=True)
