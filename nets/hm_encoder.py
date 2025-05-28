# coding=utf-8
# @Author : LJR
# @Time :  2024/10/27 20:48
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# 跳跃连接模块
class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

    # Hydra注意力机制


class HydraAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dim=None,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim if embed_dim is not None else input_dim

        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(input_dim, embed_dim) if embed_dim is not None else nn.Linear(input_dim, input_dim)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q
        batch_size, graph_size, _ = h.size()

        qkv = self.qkv(h)
        qkv = qkv.reshape(batch_size, graph_size, 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
            v = v.masked_fill(mask.unsqueeze(-1), 0)

        kv = (k * v).sum(dim=1, keepdim=True)
        out = q * kv
        out = self.proj(out)

        return out


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

        # Hydra注意力层


class HydraAttentionLayer(nn.Sequential):
    def __init__(
            self,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch'):
        super(HydraAttentionLayer, self).__init__(
            SkipConnection(
                HydraAttention(
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.GELU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphHydraEncoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphHydraEncoder, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.layers = nn.Sequential(*(
            HydraAttentionLayer(embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h = self.layers(h)
        return h


def main():
    # 模型参数
    batch_size = 32
    graph_size = 40
    node_dim = 3
    embed_dim = 128
    n_layers = 3

    # 创建模型
    model = GraphHydraEncoder(
        embed_dim=embed_dim,
        n_layers=n_layers,
        node_dim=node_dim
    )

    # 生成随机输入数据
    x = torch.randn(batch_size, graph_size, node_dim)

    # 前向传播
    output, graph_embedding = model(x)

    # 打印输出形状
    print("Output shape:", output.shape)  # [batch_size, graph_size, embed_dim]
    print("Graph embedding shape:", graph_embedding.shape)  # [batch_size, embed_dim]

    # 验证输出维度是否正确
    assert output.shape == (batch_size, graph_size, embed_dim)
    assert graph_embedding.shape == (batch_size, embed_dim)
    print("All tests passed!")


if __name__ == "__main__":
    main()