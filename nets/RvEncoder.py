import math
import torch
from torch import Tensor, nn
import revtorch as rv
from nets.MyDiffAttention import MultiHeadDiffAttention
from nets.graph_encoder import GraphAttentionEncoder, FocusedLinearAttention


class MHABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        # self.mha = nn.MultiheadAttention(hidden_size, num_heads, bias=False)
        self.mha = FocusedLinearAttention(num_heads=num_heads, dim=hidden_size)

    def forward(self, hidden_states: Tensor):

        assert hidden_states.dim() == 3
        hidden_states = self.mixing_layer_norm(hidden_states.transpose(1, 2)).transpose(
            1, 2
        )
        # hidden_states_t = hidden_states.transpose(0, 1)
        # mha_output = self.mha(hidden_states_t, hidden_states_t, hidden_states_t)
        mha_output = self.mha(hidden_states)

        return mha_output

class FFBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: Tensor):
        hidden_states = (
            self.output_layer_norm(hidden_states.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
        )
        intermediate_output = self.feed_forward(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)

        return output


class RevMHAEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        embedding_dim: int,
        input_dim: int,
        intermediate_dim: int,
        add_init_projection=True,
    ):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.num_hidden_layers = n_layers
        blocks = []
        for id in range(n_layers):
            f_func = MHABlock(embedding_dim, n_heads)
            # f_func = FocusedLinearAttention(num_heads=n_heads, dim=embedding_dim)
            # f_func = MultiHeadDiffAttention(n_embd=embedding_dim, n_head=n_heads//2, layer_idx=id+1)

            g_func = FFBlock(embedding_dim, intermediate_dim)
            # we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func, split_along_dim=-1))

        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x: Tensor, mask=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)
        x = torch.cat([x, x], dim=-1)
        out = self.sequence(x)
        return torch.stack(out.chunk(2, dim=-1))[-1]


def main():
    # ... [你的参数设置和模型创建代码保持不变]
    # [原有的模型参数设置...]
    n_layers = 3
    n_heads = 8
    embedding_dim = 128
    input_dim = 128
    intermediate_dim = 512
    batch_size = 32
    sequence_length = 40

    '''
    Using device: cuda
Initial GPU memory allocated: 2399744
Initial GPU memory reserved: 4194304

Testing forward pass...
Input shape: torch.Size([32, 40, 3])
Output shape: torch.Size([32, 40, 128])
GPU memory allocated: 3710464
GPU memory reserved: 29360128
Model parameters memory: 0.00 MB
Activations memory: 0.00 MB
    '''

    # 创建模型
    # model = RevMHAEncoder(
    #     n_layers=n_layers,
    #     n_heads=n_heads,
    #     embedding_dim=embedding_dim,
    #     input_dim=input_dim,
    #     intermediate_dim=intermediate_dim,
    #     add_init_projection=True
    # )

    model = GraphAttentionEncoder(
        n_heads=8,
        embed_dim=128,
        n_layers=3,
        normalization='batch',
        feed_forward_hidden=512
    )

    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # 创建示例输入数据
    x = torch.randn(batch_size, sequence_length, input_dim).to(device)

    # 在进行前向传播之前打印初始显存占用
    if device.type == 'cuda':
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device)}")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(device)}")

        # 测试前向传播
    print("\nTesting forward pass...")
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")

        # 在前向传播之后打印显存占用
        if device.type == 'cuda':
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device)}")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device)}")

            # 计算模型参数占用的显存（以MB为单位）
            model_memory = sum([param.nelement() * param.element_size() for param in model.parameters()]) / (
                        1024 ** 3)
            print(f"Model parameters memory: {model_memory:.2f} MB")

            # 计算激活值占用的显存
            activations_memory = (torch.cuda.memory_allocated(device) - torch.cuda.memory_allocated(device)) / (
                        1024 ** 3)
            print(f"Activations memory: {activations_memory:.2f} MB")

    print("\nModel test completed successfully!")

if __name__ == "__main__":
    main()

