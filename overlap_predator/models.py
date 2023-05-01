from kpconv.blocks import *
import torch.nn.functional as F
import numpy as np
from utils import square_distance
from utils import to_o3d_pcd, fpfh_calculate
from copy import deepcopy


def get_graph_feature(coords, feats, k=10):
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    dist = square_distance(coords.transpose(1, 2), coords.transpose(1, 2))

    idx = dist.topk(k=k + 1, dim=-1, largest=False, sorted=True)[
        1]  # [B, N, K+1], here we ignore the smallest element as it's the query itself
    idx = idx[:, :, 1:]  # [B, N, K]

    idx = idx.unsqueeze(1).repeat(1, C, 1, 1)  # [B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1, index=idx)  # [B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1, 1, 1, k)

    feats_cat = torch.cat((feats, neighbor_feats - feats), dim=1)

    return feats_cat


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, k=10):
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)

        self.conv2 = nn.Conv2d(feature_dim * 2, feature_dim * 2, kernel_size=1, bias=False)
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        self.k = k

    def forward(self, coords, features):
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  # [B, C, N, 1]

        x1 = get_graph_feature(coords, x0.squeeze(-1), self.k)
        x1 = F.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
        x1 = x1.max(dim=-1, keepdim=True)[0]

        x2 = get_graph_feature(coords, x1.squeeze(-1), self.k)
        x2 = F.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = torch.cat((x0, x1, x2), dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        return x3


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class GCN(nn.Module):
    """
        Alternate between self-attention and cross-attention
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """

    def __init__(self, num_head: int, feature_dim: int, k: int, layer_names: list):
        super().__init__()
        self.layers = []
        for atten_type in layer_names:
            if atten_type == 'cross':
                self.layers.append(AttentionalPropagation(feature_dim, num_head))
            elif atten_type == 'self':
                self.layers.append(SelfAttention(feature_dim, k))
        self.layers = nn.ModuleList(self.layers)
        self.names = layer_names

    def forward(self, coords0, coords1, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                # desc0 = desc0 + checkpoint.checkpoint(layer, desc0, desc1)
                # desc1 = desc1 + checkpoint.checkpoint(layer, desc1, desc0)
                desc0 = desc0 + layer(desc0, desc1)
                desc1 = desc1 + layer(desc1, desc0)
            elif name == 'self':
                desc0 = layer(coords0, desc0)
                desc1 = layer(coords1, desc1)
        return desc0, desc1


class KPFCNN(nn.Module):

    def __init__(self, config):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim
        self.K = config.num_kernel_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.final_feats_dim = config.final_feats_dim
        self.condition = config.condition_feature
        self.add_cross_overlap = config.add_cross_score

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # bottleneck layer and GNN part
        #####################
        gnn_feats_dim = config.gnn_feats_dim
        self.bottle = nn.Conv1d(in_dim, gnn_feats_dim, kernel_size=1, bias=True)
        k = config.dgcnn_k
        num_head = config.num_head
        self.gnn = GCN(num_head, gnn_feats_dim, k, config.nets)
        self.proj_gnn = nn.Conv1d(gnn_feats_dim, gnn_feats_dim, kernel_size=1, bias=True)
        self.proj_score = nn.Conv1d(gnn_feats_dim, 1, kernel_size=1, bias=True)

        #####################
        # List Decoder blocks
        #####################
        if self.add_cross_overlap:
            out_dim = gnn_feats_dim + 2
        else:
            out_dim = gnn_feats_dim + 1

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
        return

    def regular_score(self, score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    def forward(self, batch):
        # Get input features
        x = batch['features'].clone().detach()
        len_src_c = batch['stack_lengths'][-1][0]
        len_src_f = batch['stack_lengths'][0][0]
        pcd_c = batch['points'][-1]
        pcd_f = batch['points'][0]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]

        sigmoid = nn.Sigmoid()
        #################################
        # 1. joint encoder part
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        #################################
        # 2. project the bottleneck features
        feats_c = x.transpose(0, 1).unsqueeze(0)  # [1, C, N]
        feats_c = self.bottle(feats_c)  # [1, C, N]
        unconditioned_feats = feats_c.transpose(1, 2).squeeze(0)

        #################################
        # 3. apply GNN to communicate the features and get overlap score
        src_feats_c, tgt_feats_c = feats_c[:, :, :len_src_c], feats_c[:, :, len_src_c:]
        src_feats_c, tgt_feats_c = self.gnn(src_pcd_c.unsqueeze(0).transpose(1, 2),
                                            tgt_pcd_c.unsqueeze(0).transpose(1, 2), src_feats_c, tgt_feats_c)
        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=-1)

        feats_c = self.proj_gnn(feats_c)
        scores_c = self.proj_score(feats_c)

        feats_gnn_norm = F.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0, 1)  # [N, C]
        feats_gnn_raw = feats_c.squeeze(0).transpose(0, 1)
        scores_c_raw = scores_c.squeeze(0).transpose(0, 1)  # [N, 1]

        ####################################
        # 4. decoder part
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0, 1))

        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]

        temperature = torch.exp(self.epsilon) + 0.03
        s1 = torch.matmul(F.softmax(inner_products / temperature, dim=1), tgt_scores_c)
        s2 = torch.matmul(F.softmax(inner_products.transpose(0, 1) / temperature, dim=1), src_scores_c)
        scores_saliency = torch.cat((s1, s2), dim=0)

        if (self.condition and self.add_cross_overlap):
            x = torch.cat([scores_c_raw, scores_saliency, feats_gnn_raw], dim=1)
        elif (self.condition and not self.add_cross_overlap):
            x = torch.cat([scores_c_raw, feats_gnn_raw], dim=1)
        elif (not self.condition and self.add_cross_overlap):
            x = torch.cat([scores_c_raw, scores_saliency, unconditioned_feats], dim=1)
        elif (not self.condition and not self.add_cross_overlap):
            x = torch.cat([scores_c_raw, unconditioned_feats], dim=1)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        feats_f = x[:, :self.final_feats_dim]
        scores_overlap = x[:, self.final_feats_dim]
        scores_saliency = x[:, self.final_feats_dim + 1]

        # safe guard our score
        scores_overlap = torch.clamp(sigmoid(scores_overlap.view(-1)), min=0, max=1)
        scores_saliency = torch.clamp(sigmoid(scores_saliency.view(-1)), min=0, max=1)
        scores_overlap = self.regular_score(scores_overlap)
        scores_saliency = self.regular_score(scores_saliency)

        # normalise point-wise features
        feats_f = F.normalize(feats_f, p=2, dim=1)
        feats_f = torch.from_numpy(np.concatenate([fpfh_calculate(to_o3d_pcd(pcd_f[:len_src_f]), radius_normal=0.05, radius_feature=0.3), fpfh_calculate(to_o3d_pcd(pcd_f[len_src_f:]), radius_normal=0.05, radius_feature=0.3)], axis=0)).float().to(feats_f.device)

        return feats_f, scores_overlap, scores_saliency


class Config:
    def __init__(self):
        self.num_layers = 3
        self.in_points_dim = 3
        self.first_feats_dim = 512
        self.final_feats_dim = 96
        self.first_subsampling_dl = 0.06
        self.in_feats_dim = 1
        self.conv_radius = 2.75
        self.deform_radius = 5.0
        self.num_kernel_points = 15
        self.KP_extent = 2.0
        self.KP_influence = "linear"
        self.aggregation_mode = "sum"
        self.fixed_kernel_points = "center"
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.02
        self.deformable = False
        self.modulated = False
        self.add_cross_score = True
        self.condition_feature = True

        self.gnn_feats_dim = 256
        self.dgcnn_k = 10
        self.num_head = 4
        self.nets = ['self', 'cross', 'self']

        self.architecture = [
            'simple',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'nearest_upsample',
            'unary',
            'unary',
            'nearest_upsample',
            'unary',
            'last_unary'
        ]


def get_model():
    config = Config()
    model = KPFCNN(config).to(torch.device("cuda:0"))
    model.load_state_dict(torch.load("../overlap_predator/PREDATOR.pth")["state_dict"])
    return model
