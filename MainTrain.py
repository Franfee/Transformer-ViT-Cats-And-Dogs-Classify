# -*- coding: utf-8 -*-
# @Time    : 2022/12/31 12:15
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from net_env.net_env import *
from network.vit import ViT
from util.dataLoader import getDataLoader
from util.seed_everything import seed_everything

# ==========================================================
# ----------seed-------------
seed_everything(SEED)

# ---------switch-------------
TRAIN = True
TEST = False                                                # no output
VISDOM = False                                              # not extended implementation

# ---------dataset-----------
if TRAIN:
    TRAIN_LOADER, VALID_LOADER = getDataLoader('train')     # with label
if TEST:
    TEST_LOADER = getDataLoader('test')                     # no label
# --------visual train--------
if VISDOM:
    from visdom import Visdom
    VIZ = Visdom()
# ==========================================================


def Train(net, criterion, optimizer, scheduler):
    print("In Train...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_accuracy = 0
        with tqdm(total=len(TRAIN_LOADER), ncols=100) as t:
            t.set_description(f"EPOCH {epoch + 1}/{EPOCHS}")
            for data, label in TRAIN_LOADER:
                t.update()  # 更新进度条显示

                data = data.to(DEVICE)
                label = label.to(DEVICE)

                logist = net(data)
                loss = criterion(logist, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (logist.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(TRAIN_LOADER)
                epoch_loss += loss / len(TRAIN_LOADER)
            # end all batch
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in VALID_LOADER:
                    data = data.to(DEVICE)
                    label = label.to(DEVICE)

                    val_output = net(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(VALID_LOADER)
                    epoch_val_loss += val_loss / len(VALID_LOADER)
            print(f"Epoch {epoch + 1}/{EPOCHS}: - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} "
                  f"- val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
            # end one epoch
        # end one epoch tqdm
        # scheduler.step()是放在每个batch计算完loss并反向传播更新梯度之后
        # optimizer.step()应该在train()里面的(每batch-size更新一次梯度)
        scheduler.step()
    # end all epoch
    torch.save(net.state_dict(), MODEL_PATH)
    print("model saved.")


def Test(net):
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("model loaded.")
    result = []
    with torch.no_grad():
        for data, _ in TEST_LOADER:
            data = data.to(DEVICE)
            test_output = net(data)
            result.append(test_output.argmax(dim=1).detech().numpy())
    # print(result)
    print("test over")


if __name__ == '__main__':
    # Visual Transformer
    model = ViT(image_size=224, patch_size=32, num_classes=2, dim=128, depth=12, heads=12, mlp_dim=128).to(DEVICE)

    # loss function
    criterionFun = nn.CrossEntropyLoss()
    # optimizer
    optimizerFun = optim.Adam(model.parameters(), lr=LR_RATE)
    # scheduler
    schedulerFun = StepLR(optimizerFun, step_size=1, gamma=GAMMA)

    if TRAIN:
        Train(model, criterionFun, optimizerFun, schedulerFun)
    if TEST:
        clone = ViT(image_size=224, patch_size=32, num_classes=2, dim=128, depth=12, heads=12, mlp_dim=128).to(DEVICE)
        # 载入网络参数
        Test(clone)
