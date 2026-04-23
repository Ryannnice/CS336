 # 导入 PyTorch 主包，后面会用到张量类型标注和模块操作。
import torch

 # 导入 PyTorch 的神经网络模块基类 nn.Module。
import torch.nn as nn

 # 导入前面在 modules.py 里实现的基础组件，例如 Embedding、RMSNorm、MHA、Linear。
import cs336_basics.model.modules as modules

 # 给 SwiGLU 起一个更短的别名 FFN，表示这里把它当作前馈网络来用。
from  cs336_basics.model.modules import SwiGLU as FFN


"""
标准的 Pre-Norm Transformer block: 

- 先归一化
- 再做注意力
- 再残差相加
- 再归一化
- 再做前馈网络
- 再残差相加

输入输出 shape 都是：(batch, seq, d_model)
"""
# 定义一个 Transformer block。
# 这是整个 Transformer 语言模型的基本重复单元。
class transformer_block(nn.Module):
    # 初始化一个 Transformer block。
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        # 下面的文档字符串说明 block 的核心超参数。
        """
        d_model: int  Dimensionality of the Transformer block inputs.
        num_heads: int  Number of heads to use in multi-head self-attention.
        d_ff: int  Dimensionality of the position-wise feed-forward inner layer.
        """
        # 调用父类构造函数，把当前类注册为标准神经网络模块。
        super().__init__()

        # 保存隐藏维度，也就是每个 token 表示向量的长度。
        self.d_model = d_model

        # 保存注意力头数。
        self.num_heads = num_heads

        # 保存前馈层内部维度。
        self.d_ff = d_ff
        # d_ff 是 前馈网络内部的隐藏维度。token 向量先从 d_model 扩到 d_ff，再缩回 d_model
        # 这里用的不是最朴素的两层 MLP，而是 SwiGLU，但本质还是“一个中间隐藏宽度 d_ff”。

        # 定义第一个 RMSNorm。
        # 它作用在注意力层之前，所以是 Pre-Norm 结构的一部分。
        self.ln1 = modules.RMSNorm(d_model)

        # 定义第二个 RMSNorm。
        # 它作用在前馈网络之前。
        self.ln2 = modules.RMSNorm(d_model)

        # 定义多头自注意力子层。
        # 这里传入 max_seq_len 和 theta，是为了在注意力内部启用 RoPE。
        self.attn = modules.multihead_self_attention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta, # 旋转频率
        )

        # 下面这一段是为了兼容测试的 state_dict 加载方式。
        # RoPE 的 inv_freq 是 buffer，不是可训练参数。
        # 这里把它注册成 non-persistent，避免严格加载权重时要求外部必须提供它。
        if hasattr(self.attn, "pe") and hasattr(self.attn.pe, "inv_freq"):
            # 先把当前的 inv_freq 取出来暂存。
            buf = self.attn.pe.inv_freq

            # 尝试把它重新注册成 non-persistent buffer。
            try:
                # 先从原始 buffer 表里删掉。
                del self.attn.pe._buffers["inv_freq"]

                # 再以 persistent=False 的形式注册回去。
                self.attn.pe.register_buffer("inv_freq", buf, persistent=False)
            except Exception:
                # 如果这里失败，就保持原状，不影响正常前向传播。
                pass

        # 定义前馈网络子层。
        # 这里使用的是 SwiGLU 版本的 FFN。
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        # 中间隐藏层只有一层。维度变化：d_model -> d_ff(=128) -> d_model

        # 如果构造时显式指定了 device 或 dtype，就把整个模块移动过去。
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    # 定义 block 的前向传播。
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        # 下面的文档字符串说明输入形状和 block 的数学结构。
        """
        Input tensor shape: (batch, seq_len, d_model)
        Pre-norm Transformer block:
            x = x + Attn( LN1(x) )
            x = x + FFN( LN2(x) )
        """
        # 先把输入保存到 x，便于后面逐步更新。
        x = in_features

        # 第一步：Pre-Norm 注意力子层
        # 先做 ln1，再送进 attn，最后和原来的 x 做残差相加
        x = x + self.attn(self.ln1(x))

        # 第二步：Pre-Norm 前馈子层
        # 先做 ln2，再送进 ffn，最后再次做残差相加
        x = x + self.ffn(self.ln2(x))

        # 返回当前 block 的输出。
        return x
    

# 定义整个 Transformer 语言模型。完整的
# 它会把 token id 输入映射成对整个词表的 logits 输出。
class transformer_lm(nn.Module):
    # 初始化整个语言模型。
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, rope_theta: float, d_ff: int):
        # 下面的文档字符串说明模型级别的主要超参数。
        '''
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
        embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality of
        the position embedding matrix.
        num_layers: int The number of Transformer blocks to use.
        '''
        # 调用父类构造函数，把当前类注册为标准神经网络模块。
        super().__init__()

        # 保存词表大小。
        self.vocab_size = vocab_size

        # 保存最大上下文长度。
        self.context_length = context_length

        # 保存 Transformer block 的层数。
        self.num_layers = num_layers

        # 定义 token embedding 层。
        # 输入 token id，输出形状为 (batch, seq, d_model) 的向量表示。
        self.token_embedding = modules.Embedding(vocab_size, embedding_dim=d_model)

        # 定义多个 Transformer block，并用 ModuleList 保存。
        # 这表示模型会把输入依次送过 num_layers 个 block。
        self.layers = nn.ModuleList(
            [
                # 创建单个 block，并把对应超参数传进去。
                transformer_block(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff, # d_model -> d_ff(=128) -> d_model
                    max_seq_len=context_length,
                    theta=rope_theta # 'rope_theta': 10000.0 
                )
                for _ in range(num_layers) # 一共创建了num_layers个独立的transformer_block（而不是基于相同地址的复制）
            ]
        )

        # 定义最后输出前的 RMSNorm。
        # 作用是让最终隐藏状态在投影到词表前更稳定。
        self.output_norm = modules.RMSNorm(d_model)

        # 定义输出投影层。
        # 它把最后的隐藏表示从 d_model 维映射到 vocab_size 维，
        # 也就是对词表中每个 token 打一个分数。
        self.output_embedding = modules.Linear(in_features=d_model, out_features=vocab_size)
        # modules.Linear 的核心：einsum(x, self.weight, '... d_in,  d_out d_in -> ... d_out')




    # 定义整个语言模型的前向传播 !!!!
    def forward(self, x: torch.Tensor):
        # 先把输入的 token id 查表变成 token 向量。
        # 形状从 (batch, seq) 变成 (batch, seq, d_model)。
        x = self.token_embedding(x)

        # 依次通过每一层 Transformer block。
        for layer in self.layers:
            x = layer(x) # TRANSFORMER BLOCKS ARE AT HERE

        # 经过所有 block 后，再做一次输出归一化。
        x = self.output_norm(x)

        # 把最后的隐藏状态映射到词表大小，得到每个位置的 logits。
        x = self.output_embedding(x)

        # 返回 logits。
        # 这里不做 softmax，因为训练时 cross entropy 会自己处理 logits。
        return x
        """
        x: (B, S, vocab_size)
        - 第 1 维 B: 第几个样本
        - 第 2 维 S: 样本中的第几个 token 位置
        - 第 3 维 V: 对词表里每个 token 的打分
        """
