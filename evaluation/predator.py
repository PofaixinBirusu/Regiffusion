import numpy as np
import open3d as o3d
import torch
from overlap_predator.models import *
from overlap_predator.dataloader import get_dataloader
from dataset import PokemonTest
from utils import ransac_pose_estimation, compute_metrics, to_o3d_pcd
from models import DDPM, VoxelSuperResolution
from torch.utils import data


net = get_model()
net.eval()
device = torch.device("cuda:0")

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
# Predator is a RANSAK based method, and the final results have significant randomness

    if use_regiffusion:
        class RegiffusionWrapperDataset(data.Dataset):
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, index):
                src, tgt, T, raw = self.ds[index]
                with torch.no_grad():
                    x, voxel = diffusion(None, None, src[:1024, :].to(device).unsqueeze(0), tgt[:1024, :].to(device).unsqueeze(0))
                    new_tgt, voxel_point = sr(voxel.unsqueeze(0), None)
                if overlap_rate <= 0:
                    pass
                    tgt_ind = square_distance(new_tgt, tgt[:1024, :].to(device).unsqueeze(0))[0].min(dim=1)[0] > 0.01
                    new_tgt = new_tgt[:, tgt_ind]
                else:
                    tgt_ind = square_distance(new_tgt, tgt.to(device).unsqueeze(0))[0].min(dim=1)[0] > 0.01
                    new_tgt = torch.cat([new_tgt[:, tgt_ind], tgt.to(device).unsqueeze(0)], dim=1)
                new_tgt = new_tgt[0].cpu()
                # tgt = tgt[torch.LongTensor(np.random.permutation(tgt.shape[1])[:2048])]
                return src, new_tgt, T, raw

        if overlap_rate == 0:
            test_dataset = RegiffusionWrapperDataset(PokemonTest("zero", root_path=".."))
        elif overlap_rate > 0:
            test_dataset = RegiffusionWrapperDataset(PokemonTest("partial", overlap_rate=overlap_rate, root_path=".."))
        else:
            test_dataset = RegiffusionWrapperDataset(PokemonTest("negative", root_path=".."))

    else:
        if overlap_rate == 0:
            test_dataset = PokemonTest("zero", root_path="..")
        elif overlap_rate > 0:
            test_dataset = PokemonTest("partial", overlap_rate=overlap_rate, root_path="..")
        else:
            test_dataset = PokemonTest("negative", root_path="..")
    loader, _ = get_dataloader(test_dataset, batch_size=1, num_workers=0, shuffle=False, config=Config(), neighborhood_limits=[40, 40, 40])

    n_points = 700

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

    num_iter = int(len(loader.dataset) // loader.batch_size)
    c_loader_iter = loader.__iter__()
    with torch.no_grad():
        for i in range(1, 1+num_iter):
            cnt += 1
            inputs = c_loader_iter.next()
            ##################################
            # load inputs to device.
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.to(device) for item in v]
                else:
                    inputs[k] = v.to(device)
            ###############################################
            # forward pass
            feats, scores_overlap, scores_saliency = net(inputs)  # [N1, C1], [N2, C2]
            pcd = inputs['points'][0]
            len_src = inputs['stack_lengths'][0][0]
            T = inputs['T']
            raw = inputs['raw']

            src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
            src, tgt = torch.clone(src_pcd), torch.clone(tgt_pcd)
            src_feats, tgt_feats = feats[:len_src], feats[len_src:]
            # print(src_pcd.shape, tgt_pcd.shape, src_feats.shape, tgt_feats.shape)

            src_overlap, src_saliency = scores_overlap.detach().cpu()[:len_src], scores_saliency.detach().cpu()[:len_src]
            tgt_overlap, tgt_saliency = scores_overlap.detach().cpu()[len_src:], scores_saliency.detach().cpu()[len_src:]

            src_scores = src_overlap * src_saliency
            tgt_scores = tgt_overlap * tgt_saliency

            if (src_pcd.size(0) > n_points):
                idx = np.arange(src_pcd.size(0))
                probs = (src_scores / src_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size=n_points, replace=False, p=probs)
                src_pcd, src_feats = src_pcd[idx], src_feats[idx]
            if (tgt_pcd.size(0) > n_points):
                idx = np.arange(tgt_pcd.size(0))
                probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size=n_points, replace=False, p=probs)
                tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

            ransac_T, icp_T = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, distance_threshold=0.05, mutual=False)
            metrics = compute_metrics(
                src.unsqueeze(0), tgt.unsqueeze(0), raw.unsqueeze(0),
                torch.Tensor(ransac_T[:3]).unsqueeze(0).to(device), T.unsqueeze(0)
            )
            # print(metrics)
            metrics = compute_metrics(
                src.unsqueeze(0), tgt.unsqueeze(0), raw.unsqueeze(0),
                torch.Tensor(icp_T[:3]).unsqueeze(0).to(device), T.unsqueeze(0)
            )
            src_pcd, tgt_pcd = to_o3d_pcd(src, [1, 0.706, 0]), to_o3d_pcd(tgt, [0, 0.651, 0.929])
            # print(metrics)
            # o3d.visualization.draw_geometries([src_pcd.transform(icp_T), tgt_pcd], width=1000, height=800, window_name="registration")

            r_mse, r_mae, t_mse, t_mae = r_mse + metrics['r_mse'][0], r_mae + metrics['r_mae'][0], t_mse + metrics['t_mse'][0], t_mae + metrics['t_mae'][0]
            rre, rte, chamfer_dist = rre + metrics['err_r_deg'][0], rte + metrics['err_t'][0], chamfer_dist + metrics['chamfer_dist'][0]
            # o3d.visualization.draw_geometries([src_pcd.transform(icp_T), tgt_pcd], width=1000, height=800, window_name="registration")
            print(
                "%d / %d  r_mse: %.8f  r_mae: %.8f  t_mse: %.8f  t_mae: %.5f  rre: %.8f  rte: %.5f  chamfer dist: %.8f" % (
                    cnt, len(test_dataset), r_mse / cnt, r_mae / cnt, t_mse / cnt, t_mae / cnt, rre / cnt, rte / cnt,
                    chamfer_dist / cnt
                ))