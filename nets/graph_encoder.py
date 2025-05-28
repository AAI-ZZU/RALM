import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)  # skip connection


class LocalFeatureExtractor(nn.Module):
    def __init__(self, dim, k_neighbors=10):
        super().__init__()
        self.dim = dim
        self.k = k_neighbors

        # W1: 直接映射原始特征
        # self.W1 = nn.Linear(dim, dim)

        # CNN部分: 将K+1个特征融合为1个特征
        self.conv1d = nn.Conv1d(
            in_channels=dim,  # 输入通道数为特征维度
            out_channels=dim,  # 输出通道数保持不变
            kernel_size=k_neighbors + 1,  # 卷积核大小为K+1
            stride=k_neighbors + 1,  # 步长为K+1，确保不重叠
            padding=0  # 不需要padding
        )

        # W2: CNN输出的映射
        self.W2 = nn.Linear(dim, dim)

    def forward(self, x, adj_mat):
        """
        x: [B, N, C] 节点特征
        adj_mat: [B, N, K] K近邻矩阵
        """
        B, N, C = x.shape

        # W1*xi项
        # direct_mapping = self.W1(x)  # [B, N, C]

        # 收集K近邻特征
        neighbors = torch.gather(x.unsqueeze(2).expand(-1, -1, N, -1),
                                 dim=1,
                                 index=adj_mat.unsqueeze(-1).expand(-1, -1, -1, C))

        # 拼接中心节点和邻居
        features = torch.cat([x.unsqueeze(2), neighbors], dim=2)  # [B, N, K+1, C]
        features = features.permute(0, 1, 3, 2)  # [B, N, C, K+1]

        # 一维卷积
        conv_features = self.conv1d(features.reshape(-1, C, self.k + 1))
        conv_features = conv_features.squeeze(-1).reshape(B, N, -1)

        # W2*CNN(xi, K-neighbors)项
        cnn_mapping = self.W2(conv_features)  # [B, N, C]

        # 创建权重矩阵，初始化为零
        importance_matrix = torch.zeros((B, N, N), device=x.device)

        # 填充对角线为1
        idx = torch.arange(N, device=x.device)
        importance_matrix[:, idx, idx] = 1.0

        # 为邻居节点分配权重
        # 使用cnn_mapping作为权重
        # 将邻居权重分配到对应位置
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1)
        node_idx = torch.arange(N, device=x.device).view(1, N, 1)

        # 选取cnn_mapping的权重，并分配到邻居位置
        neighbor_weights = cnn_mapping[:, :, :self.k]  # [B, N, K]
        importance_matrix[batch_idx, node_idx, adj_mat] = neighbor_weights

        return cnn_mapping, importance_matrix

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # Scaling factor

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, input_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, input_dim]

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)  # Queries
        K = torch.matmul(hflat, self.W_key).view(shp)  # Keys
        V = torch.matmul(hflat, self.W_val).view(shp)  # Values

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)  # Output heads

        # Calculate different types of disagreement losses
        subspace_disagreement = self.disagreement_regularization(heads)
        position_disagreement = self.disagreement_regularization(attn)
        output_disagreement = self.disagreement_regularization(V)

        # Combine disagreement losses
        total_disagreement_loss = (
                subspace_disagreement +
                position_disagreement +
                output_disagreement
        ) / 3

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out, total_disagreement_loss

    def disagreement_regularization(self, heads_tensor):
        """
        计算不一致性损失

        :param heads_tensor: (n_head, batch_size, node_nums, emb/n_head)
        :return: 不一致性损失
        """
        n_heads, batch_size, node_nums, d_k = heads_tensor.size()

        # Normalize heads
        heads_flat = heads_tensor.view(n_heads * batch_size, node_nums,-1)  # (n_head * batch_size, node_nums, d_k/n_head)
        heads_norm = F.normalize(heads_flat, p=2, dim=-1)

        # Calculate similarity matrix (now (n_heads * batch_size, node_nums, node_nums))
        similarity_matrix = heads_norm @ heads_norm.transpose(1, 2)

        # Remove diagonal for inconsistency measurement
        # 创建与 similarities 形状匹配的掩码
        mask = torch.eye(node_nums, device=heads_tensor.device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads,-1, -1)
        similarities = similarity_matrix.view(batch_size, n_heads, node_nums, node_nums)
        similarities = similarities.masked_fill(mask == 1, 0.0)  # Zero out self-similarity

        # Define inconsistency as negative mean similarity
        return similarities.mean()

class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()
        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class MultiHeadAttentionLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        self.attention = MultiHeadAttention(n_heads=8, input_dim=embed_dim, embed_dim=embed_dim)

        self.norm1 = Normalization(embed_dim, normalization)

        self.feed_forward = SkipConnection(
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.GELU(),
                nn.Linear(feed_forward_hidden, embed_dim)
            ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x, adj_mat=None):
        out_put, loss = self.attention(x, adj_mat)
        x = out_put + x
        x = self.norm1(x)
        x = self.feed_forward(x)
        x = self.norm2(x)
        return x, loss

class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, n_layers, node_dim=None, normalization='batch', feed_forward_hidden=512):
        super(GraphAttentionEncoder, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.layers = nn.ModuleList(
            [MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization) for _ in range(n_layers)]
        )  # Use ModuleList for dynamic loss aggregation.

    def forward(self, x, mask=None):
        assert mask is None, "TODO mask not yet supported!"
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        total_disagreement_loss = 0
        for layer in self.layers:
            h, loss = layer(h)  # Each layer returns output and its loss
            total_disagreement_loss += loss  # Accumulate loss

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graphs, (batch_size, embed_dim)
            total_disagreement_loss  # return total disagreement loss
        )


# model = GraphAttentionEncoder(n_heads=8, embed_dim=128, n_layers=3)
# x = torch.randn(1,40,128)
# print(model(x))

class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, focusing_factor=3, k_neighbors=10):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.focusing_factor = focusing_factor

        # 分别为q, k, v定义线性变换
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # 缩放参数
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

        # 局部特征提取器
        self.phi = LocalFeatureExtractor(dim, k_neighbors)

        self.gl_loal = nn.Sequential(
            nn.Linear(dim*2, dim)
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, Q, adj_mat=None):
        """
        x: (B, N, C) 输入特征
        mask: (B, N, N) - boolean mask, True表示保留，False表示屏蔽
        """

        B, N_q, C = Q.shape

        # 线性投影
        q = self.q_proj(Q)
        k = self.k_proj(Q)
        v = self.v_proj(Q)
        x_local, importance = self.phi(v, adj_mat)

        # 非线性变换和缩放
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6

        # 动态缩放
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale

        # 优化归一化和聚焦处理
        def focused_normalization(x, factor):
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            x_normalized = x / (x_norm + 1e-6)
            x_focused = torch.pow(x_normalized, factor)
            return x_focused * x_norm

        # 使用优化的归一化和聚焦
        q = focused_normalization(q, self.focusing_factor)
        k = focused_normalization(k, self.focusing_factor)

        # 分头
        q = q.reshape(B, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        '''
        qk = q @ k.transpose(1, 2)
rank_qk = torch.linalg.matrix_rank(qk) .float().mean().item()
qk_add_1d = qk + importance
rank_qk_Add_1d = torch.linalg.matrix_rank(qk_add_1d).float().mean().item()
        '''
        # 线性注意力计算
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        # 提高 KV 的区分度
        kv = (k.transpose(-2, -1) * (N_q ** -0.5)) @ (v * (N_q ** -0.5))
        x = q @ kv * z

        # Calculate different types of disagreement losses
        local_disagreement = self.disagreement_regularization(v)
        kv_disagreement = self.disagreement_regularization(kv)

        x = x.transpose(1, 2).reshape(B, N_q, C)

        x = x + x_local
        x = x.reshape(B, self.num_heads,N_q, -1)
        output_disagreement = self.disagreement_regularization(x)
        # Combine disagreement losses
        total_disagreement_loss = (local_disagreement + output_disagreement + kv_disagreement) / 3
        x = x.reshape(B,N_q,C)
        return self.out_proj(x), total_disagreement_loss

    def disagreement_regularization(self, heads_tensor):
        """
        计算不一致性损失

        :param heads_tensor: (n_head, batch_size, node_nums, emb/n_head)
        :return: 不一致性损失
        """
        batch_size, n_heads, node_nums, d_k = heads_tensor.size()
        # Normalize heads
        heads_flat = heads_tensor.reshape(n_heads * batch_size, node_nums,-1)  # (n_head * batch_size, node_nums, d_k/n_head)
        heads_norm = F.normalize(heads_flat, p=2, dim=-1)

        # Calculate similarity matrix (now (n_heads * batch_size, node_nums, node_nums))
        similarity_matrix = heads_norm @ heads_norm.transpose(1, 2)

        # Remove diagonal for inconsistency measurement
        # 创建与 similarities 形状匹配的掩码
        mask = torch.eye(node_nums, device=heads_tensor.device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads,-1, -1)
        similarities = similarity_matrix.reshape(batch_size, n_heads, node_nums, node_nums)
        similarities = similarities.masked_fill(mask == 1, 0.0)  # Zero out self-similarity

        # Define inconsistency as negative mean similarity
        return similarities.mean()