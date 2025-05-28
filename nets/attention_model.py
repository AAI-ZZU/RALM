import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from nets.LocalAttention import LocalFeatureExtractor
from nets.RALA import GraphAttentionEncoder_2
from problems.hcvrp.hcvrp import HcvrpEnv

from nets.graph_encoder import GraphAttentionEncoder, MultiHeadAttention, MultiHeadAttentionLayer, Normalization, \
    FocusedLinearAttention
from torch.nn import DataParallel
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        # if torch.is_tensor(key) or isinstance(key, slice):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )
        # return super(AttentionModelFixed, self).__getitem__(key)


# 2D-Ptr
class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 obj,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim  # deprecated
        self.obj = obj
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_hcvrp = problem.NAME == 'hcvrp'
        self.feed_forward_hidden = 4 * embedding_dim

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.depot_token = nn.Parameter(torch.randn(embedding_dim))  # depot token
        self.init_embed = nn.Linear(3, embedding_dim)  # embed linear in customer encoder
        self.node_encoder = GraphAttentionEncoder_2(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
            feed_forward_hidden=self.feed_forward_hidden,
            k_neighbors=10
        )

        self.veh_encoder_mlp = nn.Sequential(
            nn.Linear(4, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        self.veh_encoder_self_attention = MultiHeadAttention(n_heads=n_heads, input_dim=embedding_dim,
                                                             embed_dim=embedding_dim)
        self.veh_encoder_ca_node_linear_kv = nn.Linear(embedding_dim, embedding_dim * 2)
        self.veh_encoder_ca_veh_linear_q = nn.Linear(embedding_dim, embedding_dim)  # veh-node-cross-attn w_q
        self.veh_encoder_ca_linear_o = nn.Linear(embedding_dim, embedding_dim)  # veh-node-cross-attn w_k,w_v
        self.veh_encoder_w = nn.Linear(2 * embedding_dim, embedding_dim)

        self.veh_loss = None

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, embedding_dim)))
        self.focusing_factor = 3

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.W_visited = nn.Linear(embedding_dim, embedding_dim)

        assert embedding_dim % n_heads == 0

    def pre_calculate_node(self, input):
        nhead = self.n_heads
        env = HcvrpEnv(input, scale=(1, 40, 1))
        self.veh_loss = 0
        # # embed node (depot and customer)
        node_embedding = self.init_embed(env.get_all_node_state())
        # add depot token
        node_embedding[:, 0] = node_embedding[:, 0] + self.depot_token
        # 增强特征表示
        # node_embedding = self.local_att(node_embedding, env.get_k_nearest_neighbors(k=10))
        # 前K个节点的索引
        node_embedding, _, d_loss = self.node_encoder(node_embedding, env.get_k_nearest_neighbors(k=10))
        # node_embedding, _, d_loss = self.node_encoder(node_embedding)
        # node_embedding = self.W_global(node_embedding) + self.W_local(node_embedding_local)
        bs, N, d = node_embedding.size()
        # pre-calculate the K,V of the cross-attention in vehcle encoder, avoid double calculation
        # kv = self.veh_encoder_ca_node_linear_kv(node_embedding).reshape(bs,N,nhead,-1).transpose(1,2) # bs,nhead,N,d_k*2
        kv = self.veh_encoder_ca_node_linear_kv(node_embedding)
        k, v = torch.chunk(kv, 2, -1)  # bs,nhead,n,d_k,bs,nhead,n,d_k,
        # k = torch.relu(k) + 1e-6
        scale = nn.functional.softplus(self.scale)
        # k = k / scale
        # k = self.efficient_norm_focus(k,self.focusing_factor)
        return input, node_embedding, (k, v), d_loss

    # 在ATTENTION中，使用先验知识修改attention
    def veh_encoder_cross_attention(self, veh_em, node_kv, mask=None):
        '''
        :param veh_em:
        :param node_kv:
        :param action_mask: bs,M,N
        :return:
        '''
        bs, M, d = veh_em.size()

        nhead = self.n_heads
        k, v = node_kv
        _, N, _ = k.size()
        q = self.veh_encoder_ca_veh_linear_q(veh_em).reshape(bs, M, nhead, -1).transpose(1, 2)  # bs,nhead,M,d_k

        k = k.reshape(bs, N, nhead, -1).transpose(1, 2)
        v = v.reshape(bs, N, nhead, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2) / np.sqrt(q.size(-1))  # bs,nhead,M,N

        if mask is not None:
            attn[mask.unsqueeze(1).expand(attn.size())] = -math.inf

        attn = attn.softmax(-1)  # bs,nhead,M,N
        out = attn @ v  # bs,nhead,M,d_k
        out = self.veh_encoder_ca_linear_o(out.transpose(1, 2).reshape(bs, M, -1))  # bs,M,d
        return out

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        input, node_embedding, node_kv, d_loss = self.pre_calculate_node(input)
        # input, node_embedding, veh_embedding = self.initial_em(input)
        ll, pi, veh_list, cost, d_loss = self._inner(input, node_embedding, node_kv, d_loss)

        return cost, ll, d_loss

    def _inner(self, input, node_embeddings, node_kv, d_loss):
        env = HcvrpEnv(input, scale=(1, 40, 1))
        ll, pi, veh_list, veh_loss_list = [], [], [], []
        while not env.all_finished():
            # update vehicle embeddings
            veh_embeddings = self.veh_encoder(node_embeddings, node_kv, env, d_loss)
            # select action
            veh, node, log_p = self.decoder(veh_embeddings, node_embeddings, mask=env.get_action_mask(), env=env)
            # update env
            env.update(veh, node)
            veh_list.append(veh)
            pi.append(node)
            ll.append(log_p)

        # get the final cost
        cost = env.get_cost(self.obj)
        ll = torch.stack(ll, 1)  # bs,step
        pi = torch.stack(pi, 1)  # bs,step
        veh_list = torch.stack(veh_list, 1)  # bs,step
        return ll.sum(1), pi, veh_list, cost, d_loss

    # 在ATTENTION中，使用先验知识修改attention
    def decoder(self, q_em, k_em, mask=None, env=None):
        '''
        :param q_em: Q: bs,m,d
        :param k_em: K: bs,n,d
        :param mask: bs,m,n
        :return: selected index,log_pro
        '''
        bs, m, d = q_em.size()
        _, n, _ = k_em.size()
        bs_index = torch.arange(bs, device=q_em.device)

        logits = (q_em @ k_em.transpose(1, 2) / np.sqrt(d))  # bs,m,n

        # mask优化
        mask_new = mask | env.generate_attention_mask()
        if self.tanh_clipping > 0:  # 10
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:  # True
            if mask is not None:
                logits[mask_new] = -math.inf
        # dist_score = dist_score.reshape(bs,-1)
        logits = logits.reshape(bs, -1)  # bs,M*N
        p = logits.softmax(1)  # bs,M*N
        if self.decode_type == 'greedy':
            selected = p.max(1)[1]  # bs
        else:
            selected = p.multinomial(1).squeeze(1)

        log_p = p[bs_index, selected].log()
        veh, node = selected // n, selected % n
        return veh, node, log_p

    def veh_encoder(self, node_embeddings, node_kv, env, scale):
        veh_embeddings = self.veh_encoder_mlp(env.get_all_veh_state())
        bs, N, d = node_embeddings.size()
        bs, M, d = veh_embeddings.size()
        bs_index = torch.arange(bs, device=node_embeddings.device)
        veh_node_em = node_embeddings[bs_index.unsqueeze(-1), env.veh_cur_node.clone()]  # PE:bs,M,d
        # 增强的车辆上下文编码特征
        veh_embeddings = self.veh_encoder_w(torch.cat([veh_node_em, veh_embeddings], dim=-1))
        mask = env.visited.clone()
        # depot will not be masked
        mask[:, 0] = False
        mask = mask.unsqueeze(1).expand(bs, M, N)
        # 对单个车辆进行CVRP优化
        veh_embeddings_cvrp = veh_embeddings + self.veh_encoder_cross_attention(veh_embeddings, node_kv, mask)

        # 车辆协作
        veh_embeddings = veh_embeddings + self.veh_encoder_self_attention(veh_embeddings)[0]
        veh_embeddings_hcvrp = veh_embeddings + self.veh_encoder_cross_attention(veh_embeddings, node_kv, mask)
        # 不仅考虑车辆之间的协同合作，还考虑单个的车辆的路径优化
        veh_embeddings = self.fusion_mlp(torch.cat([veh_embeddings_hcvrp, veh_embeddings_cvrp], dim=-1))

        return veh_embeddings

    def get_vehicle_node_avg_embedding(self, node_embeddings, veh_part):
        """
        获取车辆经过节点的平均特征

        Args:
        - node_embeddings (torch.Tensor): 节点特征 [bs, node_num, emb]
        - veh_part (torch.Tensor): 车辆经过节点的掩码 [bs, veh_num, node_num]

        Returns:
        - avg_embeddings (torch.Tensor): 车辆经过节点的平均特征 [bs, veh_num, emb]
        """
        # 使用掩码选择每辆车经过的节点特征
        # 使用 einsum 进行高效的加权平均
        avg_embeddings = torch.einsum('bvn,bne->bve', veh_part.float(), node_embeddings)

        # 防止除零
        bs, N, _ = node_embeddings.size()

        # 计算平均特征
        avg_embeddings = avg_embeddings / N

        return avg_embeddings

    def sample_many(self, input, batch_rep=1, iter_rep=1):

        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            None,
            self.pre_calculate_node(input),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

def similarity_enhancement_loss(veh_embeddings):
    """
    相似度增强损失
    目标:最大化相似度,同时保持区分性
    """
    # 余弦相似度
    normalized_embeddings = F.normalize(veh_embeddings, dim=-1)
    similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(-1, -2))

    # 去掉对角线
    mask = torch.eye(veh_embeddings.size(1)).bool().to(veh_embeddings.device)
    similarity_matrix.masked_fill_(mask, 0)

    # 相似度损失 - 希望相似度更高
    similarity_loss = -similarity_matrix.mean()

    # 正交约束 - 防止完全重叠
    orthogonal_loss = torch.norm(
        torch.matmul(normalized_embeddings, normalized_embeddings.transpose(-1, -2)) - torch.eye(
            veh_embeddings.size(1)).to(veh_embeddings.device)
    )

    # 综合损失
    total_loss = similarity_loss + 0.1 * orthogonal_loss

    return total_loss