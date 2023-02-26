# -*- coding: utf-8 -*-
# @Time    : 2022/12/31 12:19
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


import os
import torch
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

