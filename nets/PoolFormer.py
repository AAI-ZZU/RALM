import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class SpatialPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # 直接池化，不需要额外的坐标和需求处理
        return self.pool(x.transpose(1, 2)).transpose(1, 2).expand_as(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PoolFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.token_mixer = SpatialPooling(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        # Token Mixer (Pooling)
        x = x + self.token_mixer(self.norm1(x))

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class PoolFormerEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, depth=6):
        super().__init__()

        # 初始嵌入

        self.node_init = nn.Linear(input_dim, embed_dim)
        # 节点token
        self.depot_token = nn.Parameter(torch.randn(embed_dim)) # depot token

        self.node_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )

        # 编码器块
        self.blocks = nn.ModuleList([
            PoolFormerBlock(dim=embed_dim, mlp_ratio=4.0)
            for _ in range(depth)
        ])

        self.norm = LayerNorm(embed_dim)

    def forward(self, x):
        # 节点嵌入
        x = self.node_init(x)
        x[:, 0] = x[:, 0] + self.depot_token
        x = self.node_embedding(x)

        # 通过编码器块
        for block in self.blocks:
            x = block(x)

            # 最终归一化
        x = self.norm(x)

        return x


class PoolFormerModel(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, depth=6):
        super().__init__()

        self.encoder = PoolFormerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            depth=depth
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        # 节点编码
        encoded_nodes = self.encoder(x)

        return encoded_nodes

    # 使用示例


def main():
    # 超参数
    batch_size = 16
    num_nodes = 20

    # 随机生成数据
    x = torch.randn(batch_size, num_nodes, 3)  # 输入为(bs, node_nums, 3)

    # 模型初始化
    model = PoolFormerModel(input_dim=3, embed_dim=128, depth=6)

    # 前向传播
    node_repr = model(x)

    print("Node representations shape:", node_repr.shape)


if __name__ == "__main__":
    main()