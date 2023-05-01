import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from random import choice
from utils import PointCloudGenerator, ChamferLoss, ChamferLossP, EMD
from utils import meshgrid, square_distance, densCalc
from sklearn.metrics import precision_recall_fscore_support


class PointQuantizer(nn.Module):
    def __init__(self, w=32, device=torch.device("cuda:0")):
        super(PointQuantizer, self).__init__()
        grid = meshgrid(w)
        self.grid_points = (((grid + 1 / 2) * 2 / w) - 1).view(3, -1).contiguous().t().to(device)

    def forward(self, x):
        B, N = x.shape[0], x.shape[1]
        flat_x = x.reshape(B*N, 3)
        # (BN, )
        encoding_indices = self.get_code_indices(flat_x)
        encoding_indices, _ = torch.sort(encoding_indices.view(B, N), dim=1)
        # encoding_indices = encoding_indices.view(-1)

        voxel = torch.zeros((B, 32768)).to(self.grid_points.device)
        for i in range(B):
            # print(encoding_indices[i])
            voxel[i, encoding_indices[i]] = 1
        voxel = voxel.view(B, 1, 32, 32, 32)
        voxel = (voxel - 0.5) / 0.5 * 0.8
        return encoding_indices.view(x.shape[0], x.shape[1]), voxel

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.grid_points ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.grid_points.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pq = PointQuantizer(w=32)

    def forward(self):
        pass


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.bn2 = nn.GroupNorm(32, 64)
        self.bn3 = nn.GroupNorm(32, 64)
        self.bn4 = nn.GroupNorm(32, 128)
        self.bn5 = nn.GroupNorm(32, 1024)

    def forward(self, x):
        # B N 3
        x = x.permute([0, 2, 1])
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        # B 1024
        return x


class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        self.t_linear = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GroupNorm(32, 1024),
            nn.LeakyReLU(0.2)
        )
        self.pointnet = PointNet()
        self.point_t_linear = nn.Sequential(
            nn.Linear(1024*3, 16),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2)
        )
        self.conv1 = nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16384, 512),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 16384),
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, t, src, tgt):
        # batch x 1 x 32 x 32 x 32, batch x 512, batch x N x 3, batch x N x 3
        B, N = src.shape[0], src.shape[1]
        point_global_feature = self.pointnet(torch.cat([src, tgt], dim=0))
        point_t = self.point_t_linear(torch.cat([self.t_linear(t), point_global_feature[:B], point_global_feature[B:]], dim=1))
        # batch x 512 x 8 x 8 x 8
        x32 = self.conv1(torch.cat([point_t.view(-1, 16, 1, 1, 1).repeat([1, 1, 32, 32, 32]), self.conv(x)], dim=1))
        x64 = self.conv2(x32)
        x128 = self.conv3(x64)
        x256 = self.conv4(x128)
        x128_ = self.conv5(torch.cat([x256, self.fc(x256.view(x.shape[0], -1)).view(x.shape[0], 256, 4, 4, 4)], dim=1))
        x64_ = self.conv6(torch.cat([x128, x128_], dim=1))
        x32_ = self.conv7(torch.cat([x64, x64_], dim=1))
        x = self.conv8(torch.cat([x32, x32_], dim=1))
        return x


class DDPM(nn.Module):
    def __init__(self):
        super(DDPM, self).__init__()
        # self.denoise_net = DGCNN(k=32)
        self.denoise_net = Denoise()
        self.pq = PointQuantizer(w=32)
        T = 1000
        # at = 1-0.02t/T
        self.at = 1 - 0.02 * np.array(range(0, T + 1)) / T
        self.at_ = np.copy(self.at)
        for i in range(1, T + 1):
            self.at_[i] = self.at_[i] * self.at_[i - 1]
        # print(self.at_)
        self.loss_fn = nn.MSELoss(reduction="sum")
        # self.loss_fn = nn.SmoothL1Loss(reduction="sum")

    def forward(self, x, t, src, tgt):
        if self.training:
            # x: batch_size x 4096 x 3, t: batch_size, src: batch_size x 1024 x 3
            batch_size, pts_num = x.shape[0], x.shape[1]
            _, x = self.pq(x)
            t_embedding = get_timestep_embedding(t, 512)
            inp, z = [], []
            for i in range(x.shape[0]):
                # zi, ti = torch.randn(2048, 3).to(x.device), t[i].item()
                zi, ti = torch.randn(1, 32, 32, 32).to(x.device), t[i].item()
                at_ = self.at_[ti].item()
                inp.append(np.sqrt(at_)*x[i] + np.sqrt(1-at_)*zi)
                z.append(zi)
            # batch_size 1 x 32 x 32 x 32, batch_size 1 x 32 x 32 x 32
            inp, z = torch.stack(inp, dim=0), torch.stack(z, dim=0)
            z_pred = self.denoise_net(inp, t_embedding, src, tgt)
            # F.mse_loss(reduction="sum")
            # loss = self.loss_fn(z_pred, z)
            loss = self.loss_fn(z_pred, z) / batch_size
            return loss
        else:
            # x and t is None
            # x = torch.randn(1, 1, 32, 32, 32).to(torch.device("cuda:0"))
            pts_num = 0
            pre_tag = ""
            while pts_num < 512 or pts_num > 1600:
                z = torch.randn(1, 1, 32, 32, 32).to(torch.device("cuda:0"))
                at_ = self.at_[990]
                _, x = self.pq(tgt)
                x = np.sqrt(at_)*x + np.sqrt(1-at_)*z
                sep = 50
                step = 1000
                for i in range(sep, step + 1, sep):
                    t = step - i + sep
                    # 1 x 512
                    t_embedding = get_timestep_embedding(torch.Tensor([t, t]), 512)[0].unsqueeze(0).to(x.device)
                    z_ = self.denoise_net(x, t_embedding, src, tgt)
                    # xt-1 ~ μ: 1/√at(xt - (√1-at_ - √at*√1-at-1_)*z) , σ2: 0
                    # print(src_noise.device, src_z.device)
                    at = self.at_[t] / self.at_[t - sep]
                    x = (x - (np.sqrt(1 - self.at_[t]) - np.sqrt(at) * np.sqrt(1 - self.at_[t - sep])) * z_) / np.sqrt(at)
                    # x = (x - ((1-at)/np.sqrt(1-self.at_[t]) * z_)) / np.sqrt(at)
                    print("\r%s%d / %d" % (pre_tag, i, step), end="")
                x = x.view(-1)
                pts_num = (x > 0).sum(dim=0).item()

                tag = " finish, point num: %d" % pts_num + (", once again...  " if pts_num < 512 else ", ok !")
                pre_tag = pre_tag + "1000 / 1000 " + tag
                print("\r%s" % pre_tag, end="")
            print()
            return self.pq.grid_points[x > 0], x.view(1, 32, 32, 32)

    def make_voxel_in_sr_training(self, src, tgt):
        z = torch.randn(tgt.shape[0], 1, 32, 32, 32).to(torch.device("cuda:0"))
        at_ = self.at_[990]
        _, x = self.pq(tgt)
        x = np.sqrt(at_) * x + np.sqrt(1 - at_) * z
        sep = 50
        step = 1000
        for i in range(sep, step + 1, sep):
            t = step - i + sep
            t_embedding = get_timestep_embedding(torch.Tensor([t]*z.shape[0]), 512).to(x.device)
            z_ = self.denoise_net(x, t_embedding, src, tgt)
            # xt-1 ~ μ: 1/√at(xt - (√1-at_ - √at*√1-at-1_)*z) , σ2: 0
            at = self.at_[t] / self.at_[t - sep]
            x = (x - (np.sqrt(1 - self.at_[t]) - np.sqrt(at) * np.sqrt(1 - self.at_[t - sep])) * z_) / np.sqrt(at)
        return x

    def random_sample_t(self):
        return choice(range(1, 1001))


class VoxelSuperResolution(nn.Module):
    def __init__(self, w=32):
        super(VoxelSuperResolution, self).__init__()
        self.w = w
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16384, 512),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 16384),
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 128, kernel_size=1, stride=1, padding=0)
        )
        self.fill_cls = nn.Sequential(
            nn.Conv1d(128, 16, kernel_size=1, stride=1, bias=True),
            nn.GroupNorm(2, 16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 8, kernel_size=1, stride=1, bias=True),
            nn.GroupNorm(2, 8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 4, kernel_size=1, stride=1, bias=True),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 2, kernel_size=1, stride=1, bias=True),
            # nn.Sigmoid()
        )
        self.generator = PointCloudGenerator(
            nn.Sequential(
                nn.Conv1d(128 + 2, 64, 1), nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, 1), nn.ReLU(inplace=True),
                nn.Conv1d(64, 32, 1), nn.ReLU(inplace=True),
                nn.Conv1d(32, 32, 1), nn.ReLU(inplace=True),
                nn.Conv1d(32, 16, 1), nn.ReLU(inplace=True),
                nn.Conv1d(16, 16, 1), nn.ReLU(inplace=True),
                nn.Conv1d(16, 8, 1), nn.ReLU(inplace=True),
                nn.Conv1d(8, 3, 1)),
            rnd_dim=2, res=self.w, ops=None, normalize_ratio=1, args=None
        )
        self.n_points = 4096

        grid = meshgrid(w)
        self.grid_points = (((grid + 1 / 2) * 2 / w) - 1).view(3, -1).contiguous().t()

        self.cd = ChamferLoss()
        self.cdp5 = ChamferLossP(p=5)
        self.emd = EMD()

    def forward(self, x, target=None, dis_loss="cd"):
        # x: batch x 1 x 32 x 32 x 32, batch x N x 3
        b, w = x.shape[0], x.shape[2]
        x32 = self.conv1(x)
        x64 = self.conv2(x32)
        x128 = self.conv3(x64)
        x256 = self.conv4(x128)
        x128_ = self.conv5(torch.cat([x256, self.fc(x256.view(x.shape[0], -1)).view(x.shape[0], 256, 4, 4, 4)], dim=1))
        x64_ = self.conv6(torch.cat([x128, x128_], dim=1))
        x32_ = self.conv7(torch.cat([x64, x64_], dim=1))
        # x: batch x 128 x 32 x 32 x 32
        x = self.conv8(torch.cat([x32, x32_], dim=1))
        # print(x.shape)
        # print(self.fill_cls(x.view(b, 128, -1)).shape)
        est = self.fill_cls(x.view(b, 128, -1)).view(b, 2, w, w, w)
        dens = F.relu(est[:, 0])
        dens_cls = est[:, 1].unsqueeze(1)
        dens = dens.view(b, -1).contiguous()
        # print(dens.shape, dens_cls.shape)

        dens_s = dens.sum(-1).unsqueeze(1)
        mask = dens_s < 1e-12
        ones = torch.ones_like(dens_s)
        dens_s[mask] = ones[mask]
        dens = dens / dens_s
        dens = dens.view(b, 1, w, w, w).contiguous()

        filled = torch.sigmoid(dens_cls).round()
        # filled = (torch.sigmoid(dens_cls) > 0.2).int()
        dens_ = filled * dens

        cloud, reg = self.generator.forward_fixed_pattern(
            x, dens_, self.n_points, 2
        )
        if not self.training:
            voxel_center = []
            for i in range(b):
                voxel_center.append(self.grid_points[dens_[i].view(-1) > 0])
            voxel_center = torch.stack(voxel_center, dim=0)
            # B N 3 超分辨后的点云
            return cloud.permute([0, 2, 1]), voxel_center

        # 训练阶段计算loss
        label_to_grid = square_distance(
            target, self.grid_points.unsqueeze(0).expand_as(
                torch.empty(target.shape[0], self.grid_points.shape[0], self.grid_points.shape[1])
            ).to(x.device)
        )
        # batch x 4096
        _, min_dis_grid_ind = label_to_grid.min(dim=2)
        label_bce = torch.zeros((b, self.grid_points.shape[0])).to(x.device)
        for i in range(b):
            label_bce[i, min_dis_grid_ind[i]] = 1
        label_bce = label_bce.view(-1)
        pos_num, neg_num = label_bce.sum(dim=0).item(), self.grid_points.shape[0] - label_bce.sum(dim=0).item()
        weight = torch.ones((label_bce.shape[0], )).to(x.device)
        weight[label_bce == 1] = neg_num / pos_num
        confidence_loss = F.binary_cross_entropy_with_logits(dens_cls.contiguous().view(-1), label_bce, weight=weight, reduction="none").mean()
        # confidence_loss = focal_loss(dens_cls.view(-1), label_bce).mean()
        acc = (filled.detach().view(-1) == label_bce.detach()).sum(dim=0).item() / label_bce.shape[0]
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(label_bce.detach().cpu().numpy(),
                                                                          filled.detach().view(-1).cpu().numpy(),
                                                                          average='binary')
        cloud = cloud.permute([0, 2, 1])
        cd_p_loss = self.cdp5(cloud, target)
        if dis_loss == "cd":
            cd_loss = self.cd(cloud, target)
        else:
            cd_loss = self.emd(cloud, target)

        targetdens = densCalc(target.transpose(2, 1), self.w, normalize_ratio=1)
        # print(targetdens.sum())
        # print(targetdens.shape, dens.shape)
        density_loss = F.mse_loss(dens, targetdens, reduction='mean')

        offset_loss = torch.mean(torch.sum(reg, dim=1))

        return confidence_loss, cd_p_loss, acc, cls_precision, cd_loss, offset_loss, density_loss