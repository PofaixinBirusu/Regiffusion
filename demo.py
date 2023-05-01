import numpy as np
import open3d as o3d
import torch
from dataset import PokemonTest
from models import DDPM, VoxelSuperResolution
from utils import to_o3d_pcd
from utils import fpfh_calculate, ransac_pose_estimation, compute_metrics, densCalc
from models import PointQuantizer

device = torch.device("cuda:0")
params_save_path = "./params/point-ddpm.pth"

net = DDPM()
net.to(device)


def look_add_noise():
    pq = PointQuantizer(w=32)
    grid_points = pq.grid_points
    test_set = PokemonTest()
    # test_set = PokemonTest("negative")
    # torch.random.manual_seed(43653)
    T = 1000
    # at = 1-0.02t/T
    at = 1 - 0.02 * np.array(range(0, T + 1)) / T
    at_ = np.copy(at)
    for i in range(1, T + 1):
        at_[i] = at_[i] * at_[i - 1]

    def noising(x, t):
        z = torch.randn(x.shape).to(x.device)
        return np.sqrt(at_[t])*x + np.sqrt(1-at_[t])*z

    for i in range(20, len(test_set)):
        src, tgt, T, raw = test_set[i]
        src, tgt, T, raw = src.to(device).unsqueeze(0), tgt.to(device).unsqueeze(0), T.to(device), raw.to(device)
        with torch.no_grad():
            pcd = to_o3d_pcd(raw, [0, 1, 0])
            o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="raw")
            # voxel = densCalc(raw.unsqueeze(0).transpose(2, 1), 32, normalize_ratio=1)
            _, voxel = pq(raw.unsqueeze(0))
            voxel = voxel[0].view(-1)
            voxel = noising(voxel, 150)
            voxel_points = grid_points[voxel > 0]
            voxel_pcd = to_o3d_pcd(voxel_points, [0, 1, 0])
            # voxel_pcd.colors = o3d.Vector3dVector(np.array([[0, 1, 0]]*32768)[(voxel > 0).cpu().numpy()] * voxel[voxel > 0].view(-1, 1).cpu().numpy() * 255*2)

            o3d.visualization.draw_geometries([voxel_pcd], width=1000, height=800, window_name="src and tgt")


def look_registration():
    # 扩散模型
    net.load_state_dict(torch.load(params_save_path))
    net.eval()
    # voxel 超分辨网络
    sr = VoxelSuperResolution()
    sr.to(device)
    sr.load_state_dict(torch.load("./params/voxel-super-resolution.pth"))
    sr.eval()

    test_set = PokemonTest()
    # test_set = PokemonTest("negative")
    # torch.random.manual_seed(43653)
    for i in range(0, len(test_set)):
        src, tgt, T, raw = test_set[i]
        src, tgt, T, raw = src.to(device).unsqueeze(0), tgt.to(device).unsqueeze(0), T.to(device), raw.to(device)
        with torch.no_grad():
            pcd = to_o3d_pcd(raw, [0, 1, 0])
            o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="raw")
            src_pcd = to_o3d_pcd(src[0], [1, 0.706, 0])
            tgt_pcd = to_o3d_pcd(tgt[0], [0, 0.651, 0.929])
            o3d.visualization.draw_geometries([src_pcd, tgt_pcd], width=1000, height=800, window_name="src and tgt")
            x, voxel = net(None, None, src, tgt)
            # pcd = to_o3d_pcd(x, [0, 1, 0])
            # o3d.visualization.draw_geometries([pcd], width=1000, height=800)
            dense_point, voxel_point = sr(voxel.unsqueeze(0), None)
            # print(dense_point.shape, voxel_point.shape)

            voxel_pcd = to_o3d_pcd(voxel_point[0], [1, 0.706, 0])
            dense_pcd = to_o3d_pcd(dense_point[0], [0, 0.651, 0.929])
            # print(np.asarray(dense_pcd.points).shape[0])
            o3d.visualization.draw_geometries([voxel_pcd], width=1000, height=800, window_name="voxel")
            o3d.visualization.draw_geometries([dense_pcd], width=1000, height=800, window_name="dense")
            # o3d.visualization.draw_geometries([dense_pcd, pcd], width=1000, height=800, window_name="dense")
            src_feature = fpfh_calculate(src_pcd, radius_normal=0.05, radius_feature=0.3)
            tgt_feature = fpfh_calculate(dense_pcd, radius_normal=0.05, radius_feature=0.3)
            ransac_T, icp_T = ransac_pose_estimation(src[0], dense_point[0], src_feature, tgt_feature)
            metrics = compute_metrics(
                src, tgt, raw.unsqueeze(0),
                torch.Tensor(ransac_T[:3]).unsqueeze(0).to(device), T.unsqueeze(0)
            )
            print(metrics)
            metrics = compute_metrics(
                src, tgt, raw.unsqueeze(0),
                torch.Tensor(icp_T[:3]).unsqueeze(0).to(device), T.unsqueeze(0)
            )
            print(metrics)
            o3d.visualization.draw_geometries([src_pcd.transform(icp_T), tgt_pcd], width=1000, height=800, window_name="registration")


if __name__ == '__main__':
    look_registration()
    look_add_noise()