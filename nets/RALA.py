import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.graph_encoder import FocusedLinearAttention


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, adj_mat=None):
        if adj_mat is not None:
            return input + self.module(input, adj_mat)
        else:
            return input + self.module(input)

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
            return input

class RALA(nn.Module):
    def __init__(self, dim, num_heads, k_neighbors=10):
        super().__init__()
        # 聚焦注意力
        self.foue_att = FocusedLinearAttention(num_heads=num_heads, dim=dim, k_neighbors=10)

    def forward(self, x, adj_mat=None):
        # 聚焦注意力计算全局特征
        foatt_x, loss = self.foue_att(x, adj_mat)

        return foatt_x, loss

class MultiHeadAttentionLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
            k_neighbors=10
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        self.attention = RALA(dim=embed_dim, num_heads=n_heads, k_neighbors=k_neighbors)

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
        # 这个过程中的秩，变化很大，由原本很大的秩，现在会变得很小
        output, loss = self.attention(x, adj_mat)
        x = x + output
        x = self.norm1(x)
        x = self.feed_forward(x)
        x = self.norm2(x)
        return x, loss

class GraphAttentionEncoder_2(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
            k_neighbors=10
    ):
        super(GraphAttentionEncoder_2, self).__init__()

        # 初始化嵌入  
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.n_layers = n_layers
        # 多层注意力层  
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(
                n_heads,
                embed_dim,
                feed_forward_hidden,
                normalization,
                k_neighbors
            ) for _ in range(n_layers)
        ])

    def forward(self, x, adj_mat=None):
        # 初始嵌入  
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        all_loss = 0
        # 通过多层注意力  
        for layer in self.layers:
            h, loss = layer(h, adj_mat)
            all_loss += loss
        return (
            h,  # (batch_size, graph_size, embed_dim)  
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)  ,
            all_loss / self.n_layers
        )

    # 使用示例


# if __name__ == "__main__":
#     # 参数
#     batch_size = 32
#     num_nodes = 100
#     node_dim = 128
#     embed_dim = 128
#     n_heads = 8
#     n_layers = 3
#     k_neighbors = 10
#
#     # 随机数据
#     x = torch.randn(batch_size, num_nodes, node_dim)
#     adj_mat = torch.randint(0, num_nodes, (batch_size, num_nodes, k_neighbors))
#
#     # 创建模型
#     model = GraphAttentionEncoder_2(
#         n_heads=n_heads,
#         embed_dim=embed_dim,
#         n_layers=n_layers,
#         node_dim=node_dim,
#         k_neighbors=k_neighbors
#     )
#
#     # 前向传播
#     node_embeddings, graph_embedding = model(x, adj_mat)
#
#     print("Node embeddings shape:", node_embeddings.shape)
#     print("Graph embedding shape:", graph_embedding.shape)