import numpy as np
import open3d as o3d
import torch
from rpmnet.models import get_model
from dataset import PokemonTest
from torch.utils import data
from utils import to_o3d_pcd, compute_metrics, square_distance
from copy import deepcopy
from models import DDPM, VoxelSuperResolution


net = get_model()
# test_dataset = PokemonTest("partial", overlap_rate=0.5, root_path="..")

device = torch.device("cuda:0")
net.eval()

diffusion = DDPM()
diffusion.to(device)
diffusion.load_state_dict(torch.load("../params/point-ddpm.pth"))
diffusion.eval()
# voxel 超分辨网络
sr = VoxelSuperResolution()
sr.to(device)
sr.load_state_dict(torch.load("../params/voxel-super-resolution.pth"))
sr.eval()


if __name__ == '__main__':
########### You can modify these two parameters #######
    overlap_rate = 0                                #
    use_regiffusion = True                           #
#######################################################

    if overlap_rate == 0:
        test_dataset = PokemonTest("zero", root_path="..")
    elif overlap_rate > 0:
        test_dataset = PokemonTest("partial", overlap_rate=overlap_rate, root_path="..")
    else:
        test_dataset = PokemonTest("negative", root_path="..")
    loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    jiabos_birthday = 1997311
    yiruis_birthday = 1998913
    torch.random.manual_seed(yiruis_birthday)
    np.random.seed(jiabos_birthday)

    cnt = 0
    # metric
    r_mse, r_mae = 0, 0
    t_mse, t_mae = 0, 0
    rre, rte = 0, 0
    chamfer_dist = 0
    with torch.no_grad():
        for src, tgt, T, raw in loader:
            cnt += 1
            src, tgt, T, raw = src.to(device), tgt.to(device), T.to(device), raw.to(device)
            src_pcd, tgt_pcd = to_o3d_pcd(src[0], [1, 0.706, 0]), to_o3d_pcd(tgt[0], [0, 0.651, 0.929])

            if use_regiffusion:
                old_tgt = tgt.detach().clone()
                x, voxel = diffusion(None, None, src[:, :1024, :], tgt[:, :1024, :])
                tgt, voxel_point = sr(voxel.unsqueeze(0), None)
            if overlap_rate <= 0:
                tgt_pcd = to_o3d_pcd(tgt[0], [0, 0.651, 0.929])
            # print(src.shape, tgt.shape)

            src_pcd_, tgt_pcd_ = to_o3d_pcd(src[0]), to_o3d_pcd(tgt[0])
            o3d.estimate_normals(src_pcd_)
            o3d.estimate_normals(tgt_pcd_)
            src_normals = torch.from_numpy(np.asarray(src_pcd_.normals)).float().unsqueeze(0).to(device)
            tgt_normals = torch.from_numpy(np.asarray(tgt_pcd_.normals)).float().unsqueeze(0).to(device)

            src_inp = torch.cat([src/1.5, src_normals], dim=2)
            tgt_inp = torch.cat([tgt/1.5, tgt_normals], dim=2)
            # src_inp = torch.cat([src, src_normals], dim=2)
            # tgt_inp = torch.cat([tgt, tgt_normals], dim=2)

            if use_regiffusion:
                if overlap_rate > 0:
                    tgt_inp = tgt_inp[:, np.random.permutation(tgt_inp.shape[1])[:int(tgt_inp.shape[1]*0.7)], :]
                    src_inp = src_inp[:, np.random.permutation(src_inp.shape[1])[:int(src_inp.shape[1]*0.7)], :]
                else:
                    # tgt_inp = tgt_inp[:, np.random.permutation(tgt_inp.shape[1])[:int(tgt_inp.shape[1]*0.7)], :]
                    # src_inp = src_inp[:, np.random.permutation(src_inp.shape[1])[:int(src_inp.shape[1]*0.85)], :]
                    tgt_ind = square_distance(tgt, old_tgt)[0].min(dim=1)[0] > 0.01
                    tgt_inp = tgt_inp[:, tgt_ind, :]
                    tgt_inp = tgt_inp[:, np.random.permutation(tgt_inp.shape[1])[:int(tgt_inp.shape[1]*0.8)], :]
                    src_inp = src_inp[:, np.random.permutation(src_inp.shape[1])[:int(src_inp.shape[1]*0.8)], :]

            data = {
                "points_src": src_inp,
                "points_ref": tgt_inp
            }

            T_pred, _ = net(data, num_iter=5)
            T_pred = T_pred[-1]
            metrics = compute_metrics(
                src, tgt, raw,
                T_pred, T
            )
            # print(metrics)
            T_pred = np.concatenate([T_pred[0].cpu().numpy(), np.array([[0, 0, 0, 1]])], axis=0)
            icp_result = o3d.registration_icp(src_pcd, tgt_pcd, 0.09, init=T_pred)
            icp_result = icp_result.transformation
            metrics = compute_metrics(
                src, tgt, raw,
                torch.from_numpy(icp_result).float().to(T.device).unsqueeze(0), T
            )
            # print(metrics)
            r_mse, r_mae, t_mse, t_mae = r_mse + metrics['r_mse'][0], r_mae + metrics['r_mae'][0], t_mse + metrics['t_mse'][0], t_mae + metrics['t_mae'][0]
            rre, rte, chamfer_dist = rre + metrics['err_r_deg'][0], rte + metrics['err_t'][0], chamfer_dist + metrics['chamfer_dist'][0]
            # o3d.visualization.draw_geometries([src_pcd.transform(icp_result), tgt_pcd], width=1000, height=800, window_name="registration")
            print("%d / %d  r_mse: %.8f  r_mae: %.8f  t_mse: %.8f  t_mae: %.5f  rre: %.8f  rte: %.5f  chamfer dist: %.8f" % (
                cnt, len(test_dataset), r_mse / cnt, r_mae / cnt, t_mse / cnt, t_mae / cnt, rre / cnt, rte / cnt, chamfer_dist / cnt
            ))