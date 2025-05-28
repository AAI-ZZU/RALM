# coding=utf-8
# @Author : LJR
# @Time :  2024/11/11 12:26
# @Description :
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.rms_norm import RMSNorm


def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

class MultiHeadDiffAttention(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.lambda_init = lambda_init(layer_idx)

        # split qkv
        self.q1_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q2_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k1_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k2_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, 2 * n_embd, bias=False)  # V projects to 2 * n_embd

        self.c_proj = nn.Linear(2 * n_embd, n_embd, bias=False)
        # self.subln = nn.LayerNorm(2 * self.head_size, elementwise_affine=False)
        self.subln = RMSNorm(2 * self.head_size, eps=1e-5, elementwise_affine=True)

        # Init λ across heads
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.mixing_layer_norm = nn.BatchNorm1d(n_embd)

    def forward(self, x, k=None, mask=None):
        x = self.mixing_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        Bx, Tx, Cx = x.shape
        if k is None:
            k = x
        Bk, Tk, Ck = k.shape

        # Project x to get q1, q2, k1, k2, v
        q1 = self.q1_proj(x).view(Bx, Tx, self.n_head, self.head_size).transpose(1, 2)
        q2 = self.q2_proj(x).view(Bx, Tx, self.n_head, self.head_size).transpose(1, 2)
        k1 = self.k1_proj(k).view(Bk, Tk, self.n_head, self.head_size).transpose(1, 2)
        k2 = self.k2_proj(k).view(Bk, Tk, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(k).view(Bk, Tk, self.n_head, 2 * self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        if mask is not None:
            att1[mask.unsqueeze(1).expand(att1.size())]=-math.inf
            att2[mask.unsqueeze(1).expand(att2.size())] = -math.inf

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # Compute λ for each head separately
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att1 - lambda_full * att2

        y = torch.matmul(att, v)  # [B, n_head, T, 2 * head_size]
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(Bx, Tx, 2 * Cx)
        y = self.c_proj(y)
        return y

# MLP
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, attention_class, layer_idx):
        super().__init__()
        self.ln_1 = Normalization(n_embd)
        self.attn = attention_class(n_embd, n_head, layer_idx)
        self.ln_2 = Normalization(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            print('stdv', stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())  # [batch_size, graph_size+1, embed_dim]
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


if __name__ == '__main__':
    model = MultiHeadDiffAttention(n_head=4, n_embd=128, layer_idx=1)
    x = torch.randn(size=(32, 3,128))
    k = torch.randn(size=(32, 40, 128))
    print(model(x,k).shape)