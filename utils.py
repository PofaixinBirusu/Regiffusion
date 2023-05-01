import numpy as np
import open3d as o3d
import torch
from torch import nn
import ghalton
import math
from common.math_torch import se3
from common.torch import to_numpy
from common.math.so3 import dcm2euler
from emd_util import emdModule


def processbar(current, totle):
    process_str = ""
    for i in range(int(20 * current / totle)):
        process_str += "▉"
    while len(process_str) < 20:
        process_str += " "
    return "%s|   %d / %d" % (process_str, current, totle)


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz, colors=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pts = to_array(xyz)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
    return pcd


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def to_tensor(array):
    """
    Convert array to tensor
    """
    if not isinstance(array, torch.Tensor):
        return torch.from_numpy(array).float()
    else:
        return array


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def meshgrid(s):
    r = torch.arange(s).float()
    x = r[:, None, None].expand(s, s, s)
    y = r[None, :, None].expand(s, s, s)
    z = r[None, None, :].expand(s, s, s)
    return torch.stack([x, y, z], 0)


# 超分辨
class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, f, f_):
        if f.shape[1] == 0:
            return 0
        # f : patch_num x point_num x 4
        # f_: patch_num x M x 4
        # dis: patch_num x point_num x M
        try:
            dis = square_distance(f, f_)
            # dis = torch.sqrt(dis)
            # f2f_: patch_num x point_num   f_2f: patch_num x M
            f2f_, f_2f = dis.min(dim=2)[0], dis.min(dim=1)[0]
            # d = torch.stack([f2f_.mean(dim=1), f_2f.mean(dim=1)], dim=0).max(dim=0)[0]
            d = f2f_.mean(dim=1) + f_2f.mean(dim=1)
        except:
            print(f.shape, f_.shape)
        return d.mean()


class ChamferLossP(nn.Module):
    def __init__(self, p=5):
        super(ChamferLossP, self).__init__()
        self.p = p

    def forward(self, x, y):
        # def dist_norm(x, y, p=2, chamferdist=None):
        dis = square_distance(x, y)
        dist1, dist2 = dis.min(dim=2)[1], dis.min(dim=1)[1]
        x, y = x.transpose(2, 1), y.transpose(2, 1)
        # _, _, dist1, dist2 = chamferdist(x.transpose(2, 1).contiguous(),
        #                                  y.transpose(2, 1).contiguous())  # nearest neighbor from 1->2; 2->1
        pc1 = torch.gather(y, dim=2, index=dist1.long()[:, :].unsqueeze(1).repeat(1, 3, 1))
        pc1 = (x - pc1).norm(dim=1, p=self.p)  # .norm(p=p, dim=-1)
        pc2 = torch.gather(x, dim=2, index=dist2.long()[:, :].unsqueeze(1).repeat(1, 3, 1))
        pc2 = (y - pc2).norm(dim=1, p=self.p)  # .norm(p=p, dim=-1)
        result2 = pc1.norm(p=self.p, dim=-1) + pc2.norm(p=self.p, dim=-1)
        return result2.mean()


class EMD(nn.Module):
    def __init__(self):
        super(EMD, self).__init__()
        self.emd = emdModule()

    def forward(self, x, y):
        dis, assigment = self.emd(x, y, 0.05, 3000)
        return dis.mean()


def densSample(d, n):
    b, _, g, _, _ = d.shape
    out = []
    for i in range(b):
        N = torch.zeros(g, g, g).to(d.device)
        add = torch.ones([n]).to(d.device)
        d_ = d[i, 0, :, :, :].view(-1).contiguous()
        d_sum = d_.sum().item()
        assert (np.isfinite(d_sum))
        if d_sum < 1e-12:
            d_ = torch.ones_like(d_)
        ind = torch.multinomial(d_, n, replacement=True)
        N.put_(ind, add, accumulate=True)
        out.append(N.int())
    out = torch.stack(out, dim=0)
    return out


def densCalc(x, grid_size, normalize_ratio=1):
    n = x.size(2)
    if normalize_ratio == 0.5:
        ind = ((x + normalize_ratio) * grid_size - normalize_ratio).round().clamp(0, grid_size - 1).long()
    elif normalize_ratio == 1:
        ind = ((x + normalize_ratio) / 2 * grid_size - normalize_ratio / 2).round().clamp(0, grid_size - 1).long()
    resf1 = torch.zeros((x.size(0), grid_size ** 3)).to(x)
    ind = ind[:, 2, :] + (grid_size * ind[:, 1, :]) + (grid_size * grid_size * ind[:, 0, :])
    values = torch.ones(ind.size()).to(x)
    resf1.scatter_add_(1, ind, values)
    resf1 = resf1.reshape((-1, 1, grid_size, grid_size, grid_size)) / n
    return resf1


class PointCloudGenerator(nn.Module):
    def __init__(self, generator, rnd_dim=2, res=26, ops=None, normalize_ratio=1, args=None):
        super(self.__class__, self).__init__()

        self.base_dim = rnd_dim
        self.generator = generator
        self.ops = ops

        grid = meshgrid(res)
        if normalize_ratio == 0.5:
            self.o = (((grid + normalize_ratio) / res) - normalize_ratio).view(3, -1).contiguous()
        elif normalize_ratio == 1:
            self.o = (((grid + normalize_ratio / 2) * 2 / res) - normalize_ratio).view(3, -1).contiguous()
        self.s = res
        self.args = args

    def forward_fixed_pattern(self, x, dens, n, ratio=2):
        b, c, g, _, _ = x.shape
        grid_o = self.o.to(x.device)  # 3d meshgrid : 3* ()
        N = densSample(dens, n)
        N = N.view(b, -1).contiguous()
        x = x.view(b, c, -1).contiguous()

        b_rnd = torch.tensor(ghalton.GeneralizedHalton(2).get(n), dtype=torch.float32).t()
        b_rnd = b_rnd.to(x) * ratio - ratio / 2
        b_rnd = b_rnd.unsqueeze(0).repeat(b, 1, 1)

        N = N.view(-1).contiguous()
        x = x.permute(1, 0, 2).contiguous().view(c, -1).contiguous()

        ind = (N > 0).nonzero().squeeze(-1)
        x_ind = torch.repeat_interleave(x[:, ind], N[ind].long(), dim=-1)
        x_ind = x_ind.view(c, b, n).contiguous().permute(1, 0, 2).contiguous()
        b_inp = torch.cat([x_ind, b_rnd], dim=1)
        grid_o = grid_o.unsqueeze(0).repeat([b, 1, 1]).permute([1, 0, 2]).contiguous().view(3, -1).contiguous()
        o_ind = torch.repeat_interleave(grid_o[:, ind], N[ind].long(), dim=-1)
        o_ind = o_ind.view(3, b, n).contiguous().permute(1, 0, 2).contiguous()

        out = self.generator(b_inp)
        norm = out.norm(dim=1)
        reg = (norm - (math.sqrt(3) / (self.s))).clamp(0)  # twice the size needed to cover a gridcell
        # print(out, out.shape)
        return out + o_ind, reg

# hand-craft feature descriptor
def fpfh_calculate(pcd, radius_normal=0.01, radius_feature=0.02, compute_normals=True):
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    if compute_normals:
        o3d.estimate_normals(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # 估计法线的1个参数，使用混合型的kdtree，半径内取最多30个邻居
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.open3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature,
        max_nn=1000)
    )  # 计算FPFH特征,搜索方法kdtree
    return pcd_fpfh.data.T

# registration
def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N]
    """
    score_mat = to_array(score_mat)
    if score_mat.ndim == 2:
        score_mat = score_mat[None, :, :]

    mutuals = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):  # loop through the batch
        c_mat = score_mat[i]
        flag_row = np.zeros_like(c_mat)
        flag_column = np.zeros_like(c_mat)

        max_along_row = np.argmax(c_mat, 1)[:, None]
        max_along_column = np.argmax(c_mat, 0)[None, :]
        np.put_along_axis(flag_row, max_along_row, 1, 1)
        np.put_along_axis(flag_column, max_along_column, 1, 0)
        mutuals[i] = (flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)


def ransac_pose_estimation(src_pcd, tgt_pcd, src_feat, tgt_feat, mutual=False, distance_threshold=0.05, ransac_n=3):
    if mutual:
        if torch.cuda.device_count() >= 1:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0, 1).to(device)).cpu()
        selection = mutual_selection(scores[None, :, :])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel, col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd, corres=corrs,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd, src_feats, tgt_feats, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), ransac_n,
            [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    ransac_T = result_ransac.transformation
    icp_result = o3d.registration_icp(src_pcd, tgt_pcd, distance_threshold, init=ransac_T)
    return ransac_T, icp_result.transformation


def compute_metrics(src, tgt, raw, pred_transforms, gt_transforms):
    """Compute metrics required in the paper
    """

    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        # points_src = data['points_src'][..., :3]
        # points_ref = data['points_ref'][..., :3]
        # points_raw = data['points_raw'][..., :3]
        points_src, points_ref, points_raw = src, tgt, raw

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        # Modified Chamfer distance
        src_transformed = se3.transform(pred_transforms, points_src)
        ref_clean = points_raw
        src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_numpy(t_mse),
            't_mae': to_numpy(t_mae),
            'err_r_deg': to_numpy(residual_rotdeg),
            'err_t': to_numpy(residual_transmag),
            'chamfer_dist': to_numpy(chamfer_dist)
        }

    return metrics