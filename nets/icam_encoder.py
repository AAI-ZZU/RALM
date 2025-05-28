import math
import torch
import torch.nn as nn


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='layer'):
        super().__init__()
        if normalization == 'batch':
            self.normalizer = nn.BatchNorm1d(embed_dim)
        elif normalization == 'layer':
            self.normalizer = nn.LayerNorm(embed_dim)
        else:
            raise ValueError("Unknown normalization type")

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.transpose(1, 2)).transpose(1, 2)
        return self.normalizer(input)

class AFTFull(nn.Module):
    def __init__(self, dim_q, dim_kv, hid_dim=128, bias_dim=None):
        super().__init__()
        self.wq = nn.Linear(dim_q, hid_dim)
        self.wk = nn.Linear(dim_kv, hid_dim)
        self.wv = nn.Linear(dim_kv, hid_dim)

        if bias_dim is not None:
            self.ffnn1 = nn.Sequential(
                nn.Linear(bias_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            self.ffnn1 = None

        self.ffnn2 = nn.Linear(hid_dim, hid_dim)

    def compute_adaptation_bias(self, bias):
        if self.ffnn1 is not None:
            return self.ffnn1(bias.view(bias.size(0), -1, bias.size(-1))).view(bias.size(0), bias.size(1), -1)
        return bias

    def forward(self, x_q, x_kv, bias, mask=None):
        Q = self.wq(x_q)
        K = self.wk(x_kv)
        V = self.wv(x_kv)

        w = self.compute_adaptation_bias(bias)

        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(-1), -float('inf'))

        exp_K = torch.exp(K)
        exp_K_V = exp_K * V

        temp = torch.matmul(torch.exp(w), exp_K_V)
        weighted = temp / torch.matmul(torch.exp(w), exp_K)

        Yt = torch.sigmoid(Q) * weighted

        return self.ffnn2(Yt)


class AAFMEncoder(nn.Module):
    def __init__(self, embed_dim, dim_q, dim_kv, n_layers, bias_dim, normalization='batch', feed_forward_hidden=512):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': AFTFull(dim_q=dim_q, dim_kv=dim_kv, hid_dim=embed_dim, bias_dim=bias_dim),
                'norm1': Normalization(embed_dim, normalization),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.GELU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ),
                'norm2': Normalization(embed_dim, normalization)
            }) for _ in range(n_layers)
        ])

    def forward(self, x_q, x_kv, bias, mask=None):
        h = x_q
        for layer in self.layers:
            h = layer['norm1'](h + layer['attention'](h, x_kv, bias, mask))
            h = layer['norm2'](h + layer['ff'](h))
        return h


if __name__ == '__main__':
    model = AAFMEncoder(embed_dim=128, dim_q=128, dim_kv=128, feed_forward_hidden=512, n_layers=12,
                        normalization='batch', bias_dim=150)
    x_q = torch.randn(32, 100, 128)
    x_kv = torch.randn(32, 150, 128)
    dist = torch.randn(32, 100, 150)

    output = model(x_q, x_kv, dist)[0]
    print(output[0])