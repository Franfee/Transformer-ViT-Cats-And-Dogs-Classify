# -*- coding: utf-8 -*-
# @Time    : 2022/12/31 12:19
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import os
import sys
import torch

# ---------------------Training settings-----------------
EPOCHS = 20
BATCH_SIZE = 128
LR_RATE = 3e-5
GAMMA = 0.7
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
SEED = 123
MODEL_PATH = "model/final.mdl"
# ---------------------project dir------------------------
BASE_DIR = sys.path[0]
if "util" in BASE_DIR:
    BASE_DIR = os.path.join(BASE_DIR, '..')
