import numpy as np
import open3d as o3d
import torch
from torch.utils import data
from dataset import PokemonTrain
from models import DDPM
from utils import processbar
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda:0")

epoch = 2000
lr_update_epoch = 100
batch_size = 4
learning_rate = 0.0005
min_learning_rate = 0.000005
params_save_path = "./params/point-ddpm.pth"


train_set = PokemonTrain()
train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)

net = DDPM()
net.to(device)
optimizer = torch.optim.Adam(lr=learning_rate, params=net.parameters())
# optimizer = torch.optim.SGD(lr=learning_rate, params=net.parameters())

scaler = GradScaler()


def update_lr(optimizer, gamma=0.5):
    global learning_rate
    learning_rate = max(learning_rate*gamma, min_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print("lr update finished  cur lr: %.5f" % learning_rate)


def train():
    min_loss = 1e16
    for epoch_count in range(1, 1 + epoch):
        loss_val, processed = 0, 0
        net.train()
        optimizer.zero_grad()
        for src, tgt, T, t, raw in train_loader:
            src, tgt, T, t, raw = src.to(device), tgt.to(device), T.to(device), t.to(device), raw.to(device)
            with autocast():
                loss = net(raw, t, src, tgt)

            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_val += loss.item()
            processed += raw.shape[0]
            print("\r进度：%s  本批loss:%.5f" % (processbar(processed, len(train_loader.dataset)), loss.item()), end="")
        print("\nepoch: %d  loss: %.5f" % (epoch_count, loss_val))
        if min_loss > loss_val:
            min_loss = loss_val
            print("save...")
            torch.save(net.state_dict(), params_save_path)
            print("save finished !!!")

        if epoch_count % lr_update_epoch == 0:
            update_lr(optimizer, 0.5)


if __name__ == '__main__':
    train()