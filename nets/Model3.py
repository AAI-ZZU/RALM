# coding=utf-8
# @Author : LJR
# @Time : 2025/3/25 18:32
# @Description : 防空反导决策模型，基于图注意力机制和Transformer的防空导弹与来袭目标交互决策模型

import math
import numpy as np
import torch
import torch.nn as nn

from graph_encoder import MultiHeadAttention, GraphAttentionEncoder


class AttentionModel(nn.Module):
    def __init__(self, obs_space=None, action_space=None, num_outputs=None, model_config=None, name=None,
                 embedding_dim=128,
                 hidden_dim=512,
                 n_encode_layers=2,
                 tanh_clipping=10,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 decode_type='greedy',
                 ):
        """
        初始化防空反导模型，包括拦截导弹和来袭导弹的编码过程，以及交叉注意力解码。

        :param embedding_dim: 嵌入维度 (d_model)，Transformer每一层的特征维度
        :param hidden_dim: 用于前馈网络的隐藏层维度
        :param n_encode_layers: Transformer编码器的层数
        :param tanh_clipping: Tanh裁剪的阈值，用于限制logits的范围，默认为10
        :param mask_inner: 是否在内部使用掩码来屏蔽注意力
        :param mask_logits: 是否对logits采用掩码逻辑
        :param normalization: 正则化类型，可选 'batch' 或 'layer'
        :param n_heads: 多头注意力的头数

        """
        # TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # 保存模型参数
        self.n_heads = n_heads
        self.d_model = embedding_dim
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.n_encode_layers = n_encode_layers
        self.normalization = normalization
        self.feed_forward_hidden = hidden_dim
        self.decode_type = decode_type

        self.pro_intercept = nn.Linear(128, self.d_model)
        self.pro_incoming = nn.Linear(128, self.d_model)

        # 拦截导弹的编码器，基于图注意力机制 (Graph Attention Encoder)
        self.intercept_encoder = GraphAttentionEncoder(
            self.n_heads,
            self.d_model,
            self.n_encode_layers,
            normalization=self.normalization,
            feed_forward_hidden=self.feed_forward_hidden)

        # 拦截导弹的编码器，结构同上
        # self.intercept_encoder = MultiHeadAttention(n_heads=n_heads, input_dim=embedding_dim, embed_dim=embedding_dim)

        # 来袭导弹的编码器，结构同上
        self.incoming_encoder = MultiHeadAttention(n_heads=n_heads, input_dim=embedding_dim, embed_dim=embedding_dim)

        # 基地编码器
        self.base_encoder = nn.Linear(self.d_model, self.d_model)

        # 什么都不做的动作 一个可学习向量
        self.do_nothing = nn.Parameter(torch.randn(embedding_dim))

        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_kv = nn.Linear(self.d_model, 2 * self.d_model)
        self.linear_cross = nn.Linear(self.d_model, self.d_model)

        self.value_head = nn.Linear(self.d_model, 1)
        self._value_out = None

    def cross_attention(self, q, kv, mask=None):
        '''
        :param veh_em:
        :param node_kv:
        :param action_mask: bs,M,N
        :return:
        '''
        bs, M, d = q.size()
        nhead = self.n_heads
        _, N, _ = kv.size()

        kv = self.linear_kv(kv).reshape(bs, N, nhead, -1).transpose(1, 2)  # bs,nhead,N,d_k*2
        k, v = torch.chunk(kv, 2, -1)  # bs,nhead,n,d_k,bs,nhead,n,d_k,

        q = self.linear_q(q).reshape(bs, M, nhead, -1).transpose(1, 2)  # bs,nhead,M,d_k
        attn = q @ k.transpose(-1, -2) / np.sqrt(q.size(-1))  # bs,nhead,M,N
        if mask is not None:
            attn[mask.unsqueeze(1).expand(attn.size())] = -math.inf
        attn = attn.softmax(-1)  # bs,nhead,M,N
        out = attn @ v  # bs,nhead,M,d_k
        out = self.linear_cross(out.transpose(1, 2).reshape(bs, M, -1))  # bs,M,d
        return out

    def forward(self, input_dict, state=None, seq_lens=None):
        """
        前向过程，包括编码器和解码器，完成防空决策生成。

        :param input_dict: 输入字典，包含拦截导弹和来袭目标的特征
        :param state: 当前环境状态（未使用，可扩展）
        :param seq_lens: 序列长度，用于动态序列处理（未使用，可扩展）
        :return: 拦截导弹索引、来袭导弹索引、对应的log概率
        """
        # 从输入中提取拦截导弹和来袭导弹的嵌入向量
        obs = input_dict["obs"]
        interceptors = obs["interceptors"]  # (batch_size, N, d_model)，N为拦截导弹数量
        incomings = obs["incomings"]  # (batch_size, M, d_model)，M为来袭导弹数量
        atcion_mask = obs["action_mask"]  # 动作掩码
        success_prob = obs["success_prob"]  # 动作掩码 (batch_size, N, M)
        do_nothing_mask = True

        interceptors = self.pro_intercept(interceptors)
        incomings = self.pro_incoming(incomings)

        bs, N, _ = interceptors.size()  # 拦截导弹的批量大小、数量、特征维度
        _, M, _ = incomings.size()  # 来袭导弹的批量大小、数量、特征维度

        # 编码器：对拦截导弹和来袭导弹执行图注意力编码，提取高层次特征
        intercept_encoded = interceptors + self.intercept_encoder(interceptors)[0]  # (batch_size, N, d_model)
        incoming_encoded = incomings + self.incoming_encoder(incomings)[0]  # (batch_size, M, d_model)

        # cross attention：使用交叉注意力机制使拦截导弹与来袭导弹交互信息
        intercept_encoded = intercept_encoded + self.cross_attention(intercept_encoded, incoming_encoded)


        # 进入解码器，生成最优拦截方案
        intercept_idx, incoming_idx, log_prob = self.decoder(intercept_encoded, incoming_encoded, atcion_mask,
                                                             success_prob, do_nothing_mask)

        return intercept_idx, incoming_idx, log_prob


    def decoder(self, intercept_encoded, incoming_encoded, action_mask, success_prob, do_nothing_mask):
        """
        解码器：根据交叉注意力结果生成动作分布，并选择最优拦截方案。

        @param intercept_encoded: 拦截导弹编码向量 (batch_size, N, d_model)
        @param incoming_encoded: 来袭导弹编码向量 (batch_size, M, d_model)
        @param action_mask: 动作掩码，屏蔽一些无效的注意力 (batch_size, N, M)
        @param success_prob: 动作成功率矩阵 (batch_size, N, M)
        @return: 拦截导弹索引、来袭导弹索引、动作对应的log概率
        """
        bs, n, d = intercept_encoded.size()  # (batch_size, N, d_model)
        _, m, _ = incoming_encoded.size()  # (batch_size, M, d_model)
        bs_index = torch.arange(bs, device=intercept_encoded.device)

        # 扩展 incoming_encoded，为 “ 什么都不做 ” 添加一个特征
        do_nothing_feature = self.do_nothing.unsqueeze(0).unsqueeze(0).expand(bs, 1, d)  # (batch_size, 1, d_model)
        incoming_encoded_extended = torch.cat([do_nothing_feature, incoming_encoded],
                                              dim=1)  # (batch_size, M+1, d_model)

        # 更新 do_nothing_mask_1，为“什么都不做”添加一列 False，使其始终合法
        do_nothing_mask_1 = None
        if action_mask is not None:
            if do_nothing_mask:
                do_nothing_mask_1 = torch.zeros((bs, n, 1), dtype=torch.bool, device=action_mask.device)  # (batch_size, N, 1)
            else:
                do_nothing_mask_1 = torch.ones((bs, n, 1), dtype=torch.bool, device=action_mask.device)  # (batch_size, N, 1)
        action_mask = torch.cat([do_nothing_mask_1, action_mask], dim=2)  # (batch_size, N, M+1)

        # 扩展 success_prob，为“什么都不做”添加默认成功率（值设置为 1.0）
        success_prob_extended = torch.cat([torch.full((bs, n, 1), 1.0, device=success_prob.device), success_prob], dim=2)  # (batch_size, N, M+1)
        logits = intercept_encoded @ incoming_encoded_extended.transpose(1, 2) / np.sqrt(d)  # (batch_size, N, M+1)

        # 引入成功率对 logits 进行加权
        logits = logits * success_prob_extended

        # Tanh 裁剪 logits 值，保持数值稳定
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        if action_mask is not None:
            # 将 action_mask 中对应 True 的位置设置为 -np.inf
            logits[action_mask] = -np.inf  # 屏蔽无效动作的 logits
        # 计算动作分布 (softmax)
        logits = logits.reshape(bs, -1)  # 展平成 (batch_size, N * (M+1))
        p = logits.softmax(1)

        # 动作选择：贪婪选择或随机采样
        if self.decode_type == 'greedy':  # 贪婪选择
            selected = p.max(1)[1]
        else:  # 随机采样
            selected = p.multinomial(1).squeeze(1)
        # 计算所选动作的 log 概率
        log_p = p[bs_index, selected].log()
        # 将一维下标解析为二维索引（包括拦截索引和扩展的来袭导弹索引）
        intercept_idx, incoming_idx_ext = selected // (m + 1), selected % (m + 1) - 1
        # 返回
        return intercept_idx, incoming_idx_ext, log_p


def train_step(model, optimizer, input_dict):
    # 清空梯度
    optimizer.zero_grad()

    # 前向传播
    intercept_idx, incoming_idx, log_prob = model(input_dict)

    # 这里假设直接把 log_prob 看成损失（仅示例）
    loss = -log_prob.mean()

    # 反向传播
    loss.backward()
    optimizer.step()


def measure_gpu_memory(model, optimizer, input_dict):
    # 同步 CUDA，确保先前操作都执行完

    start_allocated = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()

    # 进行一次前向+后向的训练过程
    train_step(model, optimizer, input_dict)

    # 再次同步，确保所有操作完成


    end_allocated = torch.cuda.memory_allocated()
    end_reserved = torch.cuda.memory_reserved()

    print(f"Allocated before: {start_allocated / 1024 ** 2:.2f} MB")
    print(f"Allocated after:  {end_allocated / 1024 ** 2:.2f} MB")
    print(f"Reserved before:  {start_reserved / 1024 ** 2:.2f} MB")
    print(f"Reserved after:   {end_reserved / 1024 ** 2:.2f} MB")
    print(f"Allocated increase: {(end_allocated - start_allocated) / 1024 ** 2:.2f} MB")
    print(f"Reserved increase:  {(end_reserved - start_reserved) / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 初始化模型并移动到GPU
    model = AttentionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 2) 构造输入数据并移动到GPU
    batch_size = 100
    N = 100
    M = 50
    embedding_dim = 128

    interceptors = torch.rand(batch_size, N, embedding_dim, device=device)
    incomings = torch.rand(batch_size, M, embedding_dim, device=device)
    action_mask = torch.zeros(batch_size, N, M, device=device, dtype=torch.bool)
    success_prob = torch.rand(batch_size, N, M, device=device)

    input_dict = {
        "obs": {
            "interceptors": interceptors,
            "incomings": incomings,
            "action_mask": action_mask,
            "success_prob": success_prob
        }
    }

    # 3) 监控单次训练的显存使用
    measure_gpu_memory(model, optimizer, input_dict)