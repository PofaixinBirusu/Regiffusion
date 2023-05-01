import numpy as np
import open3d as o3d
import torch
from torch.utils import data
from utils import square_distance, to_o3d_pcd


def generate_transform(rot_mag=45.0, trans_mag=0.5):
    anglex = np.random.uniform() * np.pi * rot_mag / 180.0
    angley = np.random.uniform() * np.pi * rot_mag / 180.0
    anglez = np.random.uniform() * np.pi * rot_mag / 180.0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])
    R_ab = Rx @ Ry @ Rz
    t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

    rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
    return rand_SE3


def random_sample_and_transform(src, tgt, sample_point_num=1024):
    T = generate_transform()
    R_to_src, t_to_src = T[:, :3].T, T[:, :3].T @ -T[:, 3:]
    src = (R_to_src @ src.T + t_to_src).T
    src_rand_idx = np.random.permutation(src.shape[0])[:sample_point_num]
    tgt_rand_idx = np.random.permutation(tgt.shape[0])[:sample_point_num]
    src, tgt = src[src_rand_idx], tgt[tgt_rand_idx]

    return src, tgt, T


class PokemonTrain(data.Dataset):
    def __init__(self, overlap_rate=None):
        self.data = []
        self.raw = []
        for name in ["Charizard", "Bulbasaur", "Mew", "Nidoking", "Pikachu"]:
            self.data.append([np.load("./data/half/%s_half_up_1.npy" % name), np.load("./data/half/%s_half_down_1.npy" % name)])
            self.data.append([np.load("./data/half/%s_half_up_2.npy" % name), np.load("./data/half/%s_half_down_2.npy" % name)])
            self.data.append([np.load("./data/half/%s_half_up_3.npy" % name), np.load("./data/half/%s_half_down_3.npy" % name)])
            self.raw = self.raw + [np.asarray(o3d.io.read_point_cloud("./data/downsampled/%s-4096.pcd" % name).points)]*3
        self.T = 1000
        self.overlap_rate = overlap_rate

    def __len__(self):
        return len(self.data) * self.T

    def __getitem__(self, index):
        # t is the timestep of diffusion model
        t = index % self.T + 1
        index = index // self.T
        src, tgt, raw = self.data[index][0], self.data[index][1], self.raw[index]
        if self.overlap_rate is not None:
            overlap_points_num_half = int((src.shape[0] + tgt.shape[0]) * self.overlap_rate / 2)
            dis = square_distance(torch.from_numpy(src).float().unsqueeze(0), torch.from_numpy(tgt).float().unsqueeze(0))[0]
            src_to_tgt_min_dis = dis.min(dim=1)[0]
            src_idx = torch.topk(src_to_tgt_min_dis, largest=False, dim=0, k=overlap_points_num_half)[1].numpy()
            tgt_to_src_min_dis = dis.t().min(dim=1)[0]
            tgt_idx = torch.topk(tgt_to_src_min_dis, largest=False, dim=0, k=overlap_points_num_half)[1].numpy()
            src, tgt = np.concatenate([src, tgt[tgt_idx]], axis=0), np.concatenate([tgt, src[src_idx]], axis=0)

        src, tgt, T = random_sample_and_transform(src, tgt)
        src, tgt, T, raw = torch.from_numpy(src).float(), torch.from_numpy(tgt).float(), torch.from_numpy(T).float(), torch.from_numpy(raw).float()
        return src, tgt, T, t, raw


class PokemonTest(data.Dataset):
    def __init__(self, mode="zero", overlap_rate=None, root_path="."):
        """
        mode can be chooesed in "zero" or "negative" or "partial"
        """
        if mode == "partial":
            self.path = root_path + "/data/test/zero-overlap-rate"
        else:
            self.path = root_path + "/data/test/%s-overlap-rate" % mode
        self.overlap_rate = overlap_rate
        self.root_path = root_path

    def __len__(self):
        return 45

    def __getitem__(self, index):
        data = np.load(self.path + "/%d.npy" % (index+1))
        src = data[:1024, :]
        tgt = data[1024:2048, :]
        T = data[2048:, :].T
        raw = np.asarray(o3d.io.read_point_cloud(self.root_path + "/data/downsampled/%s-4096.pcd" % ["Charizard", "Bulbasaur", "Mew", "Nidoking", "Pikachu"][index // 9]).points)

        if self.overlap_rate is not None:
            overlap_points_num_half = int((src.shape[0] + tgt.shape[0]) * self.overlap_rate / 2)
            src_gt = (T[:, :3] @ src.T + T[:, 3:]).T
            dis = square_distance(torch.from_numpy(src_gt).float().unsqueeze(0), torch.from_numpy(tgt).float().unsqueeze(0))[0]
            src_to_tgt_min_dis = dis.min(dim=1)[0]
            src_idx = torch.topk(src_to_tgt_min_dis, largest=False, dim=0, k=overlap_points_num_half)[1].numpy()
            tgt_to_src_min_dis = dis.t().min(dim=1)[0]
            tgt_idx = torch.topk(tgt_to_src_min_dis, largest=False, dim=0, k=overlap_points_num_half)[1].numpy()
            # # 画图专用
            # src, tgt = np.concatenate([src_gt, tgt[tgt_idx]+np.random.randn(tgt_idx.shape[0], 3)*0.01], axis=0), np.concatenate([tgt, src_gt[src_idx]+np.random.randn(src_idx.shape[0], 3)*0.01], axis=0)
            src, tgt = np.concatenate([src_gt, tgt[tgt_idx]], axis=0), np.concatenate([tgt, src_gt[src_idx]], axis=0)
            # src_rand_idx = np.random.permutation(src.shape[0])[:1024]
            # tgt_rand_idx = np.random.permutation(tgt.shape[0])[:1024]
            # src, tgt = src[src_rand_idx], tgt[tgt_rand_idx]

            src = (T[:, :3].T @ (src.T - T[:, 3:])).T

        src, tgt, T, raw = torch.from_numpy(src).float(), torch.from_numpy(tgt).float(), torch.from_numpy(T).float(), torch.from_numpy(raw).float()
        return src, tgt, T, raw


if __name__ == '__main__':
    # pokemon_train = PokemonTrain(overlap_rate=0.9)
    # for i in range(0, len(pokemon_train)):
    #     src, tgt, T, t, raw = pokemon_train[i]
    #     src_pcd, tgt_pcd = to_o3d_pcd(src, [1, 0.706, 0]), to_o3d_pcd(tgt, [0, 0.651, 0.929])
    #     raw_pcd = to_o3d_pcd(raw, [0, 0.651, 0.929])
    #     o3d.draw_geometries([src_pcd, tgt_pcd], width=1000, height=800)
    #     src_pcd.transform(np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0))
    #     o3d.draw_geometries([src_pcd, tgt_pcd], width=1000, height=800)
    #     o3d.draw_geometries([raw_pcd], width=1000, height=800)

    # pokemon_test = PokemonTest("negative")
    pokemon_test = PokemonTest("partial", overlap_rate=0.5)
    for i in range(0, len(pokemon_test)):
        src, tgt, T, raw = pokemon_test[i]
        src_pcd, tgt_pcd = to_o3d_pcd(src, [1, 0.706, 0]), to_o3d_pcd(tgt, [0, 0.651, 0.929])
        raw_pcd = to_o3d_pcd(raw, [0, 0.651, 0.929])
        o3d.draw_geometries([src_pcd, tgt_pcd], width=1000, height=800)
        src_pcd.transform(np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0))
        o3d.draw_geometries([src_pcd, tgt_pcd], width=1000, height=800)
        o3d.draw_geometries([raw_pcd], width=1000, height=800)