# coding=utf-8
# @Author : LJR
# @Time :  2024/11/27 16:44
# @Description :


import torch
import torch.nn as nn


class LocalFeatureExtractor(nn.Module):
    def __init__(self, dim, k_neighbors=10):
        super().__init__()
        self.dim = dim
        self.k = k_neighbors

        # W1: 直接映射原始特征
        self.W1 = nn.Linear(dim, dim)

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
        direct_mapping = self.W1(x)  # [B, N, C]

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

        # 最终输出: hi = W1*xi + W2*CNN(xi, K-neighbors)
        out = direct_mapping + cnn_mapping

        return out