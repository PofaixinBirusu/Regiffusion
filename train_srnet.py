import numpy as np
import open3d as o3d
import torch
from torch.utils import data
from dataset import PokemonTrain
from models import DDPM, VoxelSuperResolution
from utils import processbar, to_o3d_pcd
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda:0")

epoch = 200
lr_update_epoch = 20
batch_size = 4
learning_rate = 0.00007
min_learning_rate = 0.000005
params_save_path = "./params/voxel-super-resolution.pth"


train_set = PokemonTrain()
train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)

net = VoxelSuperResolution()
net.to(device)
# net.load_state_dict(torch.load(params_save_path))
optimizer = torch.optim.Adam(lr=learning_rate, params=net.parameters())
# optimizer = torch.optim.SGD(lr=learning_rate, params=net.parameters())
ddpm = DDPM()
ddpm.to(device)
ddpm.load_state_dict(torch.load("./params/point-ddpm.pth"))
ddpm.eval()

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

        # if epoch_count <= 1:
        #     dis_loss = "cd"
        # else:
        #     dis_loss = "emd"
        dis_loss = "emd"

        loss_val, processed = 0, 0
        precision_mean, density_mean, chamfer_mean = 0, 0, 0
        net.train()
        optimizer.zero_grad()
        for src, tgt, _, _, raw in train_loader:
            src, tgt, raw = src.to(device), tgt.to(device), raw.to(device)
            with torch.no_grad():
                voxel = ddpm.make_voxel_in_sr_training(src, tgt)
            with autocast():
                confidence_loss, cd_p_loss, acc, precision, cd_loss, offset_loss, density_loss = net(voxel, raw, dis_loss)

            loss = confidence_loss * 100 + cd_p_loss * 300 + cd_loss * 10000 + offset_loss * 0.3 + density_loss * 1e10
            precision_mean += precision * raw.shape[0]
            density_mean += density_loss.item() * 1e10 * raw.shape[0]
            chamfer_mean += cd_loss.item() * 10000 * raw.shape[0]

            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_val += loss.item()
            processed += raw.shape[0]
            print(
                "\r进度：%s  本批loss:%.5f  confidence loss: %.5f  cd p loss: %.5f  acc: %.5f  precision: %.5f  offset loss: %.5f  density loss: %.5f  cd: %.5f" % (
                    processbar(processed, len(train_set)), loss.item(), confidence_loss.item() * 100,
                    cd_p_loss.item() * 1000, acc, precision, offset_loss.item(), density_loss.item() * 1e10,
                    cd_loss.item() * 1000), end=""
            )

        chamfer_mean /= len(train_set)
        precision_mean /= len(train_set)
        density_mean /= len(train_set)
        print("\nepoch:%d  loss:%.3f  precision: %.5f  density loss: %.5f  cd: %.5f" % (epoch_count, loss_val, precision_mean, density_mean, chamfer_mean))

        if min_loss > chamfer_mean:
            min_loss = chamfer_mean
            print("save...")
            torch.save(net.state_dict(), params_save_path)
            print("save finished !!!")

        if epoch_count % lr_update_epoch == 0:
            update_lr(optimizer, 0.5)


if __name__ == '__main__':
    train()