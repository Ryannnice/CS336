 # 导入 PyTorch 主包，后面会用它创建张量和参数。
import torch 

 # 导入 PyTorch 的神经网络模块基类 nn.Module，以及参数容器等工具。
import torch.nn as nn 

 # 导入 einops 里的重排与张量乘法工具，后面注意力和 RoPE 会频繁用到。
 # RoPE: 2048维度向量，进行1024次二位旋转：[v_1, v_2], [v_3, v_4], [v_5, v_6], ..., [v_2047, v_2048]
from einops import rearrange, einsum

 # 定义一个线性层类。
 # 你可以把它理解成最基础的“全连接层”或“矩阵乘法层”。
class Linear(nn.Module):
    # 初始化线性层。
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # 下面的文档字符串说明这个层的输入输出维度含义。
        '''
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        # 先调用父类构造函数，让当前类具备 nn.Module 的能力。
        super().__init__()

        # 保存输入维度，表示输入向量最后一维的长度。
        self.in_features = in_features

        # 保存输出维度，表示输出向量最后一维的长度。
        self.out_features = out_features

        # 记录参数存放在哪个设备上，比如 cpu 或 cuda。
        self.device = device

        # 记录参数采用什么数据类型，比如 float32。
        self.dtype = dtype

        # 创建可训练权重矩阵，形状是 (out_features, in_features)。
        # 这就是线性层真正要学习的参数。
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)) # 后续有Linear时，会默认初始化到这里的权重

        # 调用自定义初始化函数，为权重赋初值。
        self._init_weight()

    # 定义前向传播：给定输入 x，输出线性变换后的结果。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 einsum 实现矩阵乘法。
        # 这里的含义是：
        # 输入 x 的最后一维是 d_in，
        # 权重 weight 的形状是 (d_out, d_in)，
        # 输出的最后一维就变成 d_out。
        return einsum(x, self.weight, '... d_in,  d_out d_in -> ... d_out')

    # 定义权重初始化函数。
    def _init_weight(self):
        # 计算截断正态初始化的标准差。
        # 这是一个和输入输出维度都相关的缩放方式。
        # Xavier 初始化的目标是使得 每一层的输入输出的方差 尽量保持一致，避免梯度在反向传播过程中消失或爆炸
        std = (2 / (self.in_features + self.out_features)) ** 0.5

        # 用截断正态分布初始化权重。
        # 权重不会初始化得过大，有助于训练稳定。
        # a 和 b 分别表示 截断正态分布（Truncated Normal Distribution）中的下限和上限
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std=std, a=-3*std, b=3*std)


# 定义 Embedding 层。
# 它的作用不是做乘法，而是“查表”：输入 token id，输出对应向量。
class Embedding(nn.Module):
    # 初始化 embedding 层。
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # 下面的文档字符串解释词表大小和向量维度。
        '''
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        # 调用父类构造函数，注册为一个标准的 PyTorch 模块。
        super().__init__()

        # 记录词表大小，也就是一共有多少个 token id。
        self.num_embeddings = num_embeddings

        # 记录每个 token 对应向量的维度。
        self.embedding_dim = embedding_dim

        # 创建 embedding 权重表，形状是 (词表大小, 向量维度)。
        # 你可以把它理解成一个“大字典”，每一行对应一个 token 的向量。
        self.embed_weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # 调用初始化函数，为整张 embedding 表赋初值。
        self._init_weight()

    # 定义前向传播：输入 token id，返回对应的 embedding 向量。
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 如果输入已经是 long 类型，就直接使用。
        if token_ids.dtype == torch.long:
            pass

        # 否则强制转成 long，因为张量索引必须使用整数类型。
        else:
            token_ids = token_ids.long()

        # 直接用 token_ids 作为索引，从 embedding 表中取出对应行。
        # 这就是“查表”操作。
        return self.embed_weight[token_ids]

    # 定义 embedding 表的初始化方式。
    def _init_weight(self):
        # 用截断正态分布初始化每个 token 向量。
        nn.init.trunc_normal_(self.embed_weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

# 定义 RMSNorm 层。
# 它的作用是把向量的数值尺度拉回到更稳定的范围，
# 让训练过程不容易因为数值过大或过小而发散。
class RMSNorm(nn.Module):
    # 初始化 RMSNorm 层。
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        # 下面的文档字符串说明隐藏维度和数值稳定项的含义。
        '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        # 调用父类构造函数，把当前类注册成标准神经网络模块。
        super().__init__()

        # 保存隐藏维度大小，也就是最后一维的长度。
        self.d_model = d_model

        # 保存一个很小的正数，防止后面除以 0。
        self.eps = eps

        # 创建可学习的缩放参数，形状是 (d_model,)。
        # RMSNorm 不会像 LayerNorm 那样再额外学习 bias，
        # 这里只学习一个逐维缩放因子。
        self.g_weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        # g_weight是个一维的向量！长度为 d_model

        # 调用初始化函数，为缩放参数赋初值。
        self._init_weight()

    # 定义前向传播：输入一个向量，输出归一化后的向量。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先记录输入原本的数据类型，最后返回前要转回去。
        in_dtype = x.dtype

        # 先把输入转成 float32 来做归一化计算，
        # 这样数值更稳定，特别是在 fp16/bf16 下更安全。
        x = x.to(dtype=torch.float32)

        # 计算每个向量的 RMS。
        # 这里是沿最后一维求平方平均，再开根号。
        # keepdim=True 的目的是保留最后一维，方便后面逐元素相除。
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)

        # 先用 x / rms 完成归一化，
        # 再乘上可学习参数 g_weight，恢复模型需要的表达能力。
        out = einsum(x/rms, self.g_weight, '... d, d -> ... d')

        # 最后把输出转回输入原本的数据类型。
        return out.to(dtype=in_dtype)
    
    # 定义 RMSNorm 的参数初始化方式。
    def _init_weight(self):
        # 用截断正态分布初始化缩放参数。
        nn.init.trunc_normal_(self.g_weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

class SwiGLU(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        if dtype is None or not torch.is_floating_point(torch.empty((), dtype=dtype)):
            dtype = torch.float32
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.w1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        self._init_weight()

    def forward(self, x) -> torch.Tensor:
        a = einsum(self.w1, x, 'd_ff d_model, ... d_model -> ... d_ff')
        step1 = a*torch.sigmoid(a)
        step2 = step1 * einsum(self.w3, x, 'd_ff d_model, ... d_model -> ... d_ff')
        return einsum(self.w2, step2, 'd_model d_ff, ... d_ff -> ... d_model')
    
    def _init_weight(self):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=1.0, a=-3.0, b=3.0)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.freq_arrange = 1 / (self.theta**(torch.arange(0, self.d_k, 2).to(dtype=torch.float)/self.d_k))
        self.register_buffer(name='inv_freq', tensor=self.freq_arrange)



    # def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    #     seq_len = x.size(-2)
    #     if token_positions is None:
    #         token_positions = torch.arange(seq_len, device=x.device)
    #         token_positions = token_positions.unsqueeze(0).expand(x.size(0), seq_len)
    #     rotated_x = self.rotate_tensor(x)
    #     theta_arange = einsum(self.inv_freq, token_positions, 'd, ... s -> ... s d')
    #     cos = theta_arange.cos().repeat_interleave(2, dim=-1)
    #     sin = theta_arange.sin().repeat_interleave(2, dim=-1)
    #     return x*cos + rotated_x*sin

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        # x 可能有两种常见形状：
        # 1) (B, S, Dh)    : 单头或尚未显式拆分 head 的表示
        # 2) (B, H, S, Dh) : 多头注意力中已经拆成多个 head 的 Q/K
        # 这里的实现目标是“输入什么形状，输出就保持什么形状”，
        # 这样 RoPE 可以被独立复用，不会偷偷多引入一个维度。
        S = x.size(-2)

        # 1) 准备 token position。
        # 如果外部没有传 position，就默认当前位置是 [0, 1, 2, ..., S-1]。
        # 如果外部传了 position，则保留它原本的批次维度信息，只把 device 对齐到 x。
        # 这样既支持共享位置编码，也支持每个 batch 有不同 position 的情况。
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device)
        else:
            token_positions = token_positions.to(device=x.device)

        # 2) 计算旋转角度矩阵 theta。
        # self.inv_freq 的形状是 (Dh/2,)，代表每一对偶/奇维度共用一个频率。
        # einsum 后：
        # - 若 token_positions 是 (S,)      -> theta 形状为 (S, Dh/2)
        # - 若 token_positions 是 (B, S)   -> theta 形状为 (B, S, Dh/2)
        # 后续再把 Dh/2 扩成 Dh，与最后一维逐元素相乘。
        theta = einsum(token_positions, self.inv_freq, "... s, d -> ... s d")

        # 3) 构造 cos/sin，并把它们扩展成可与 x 广播的形状。
        # repeat_interleave(2) 的原因是：
        # 每一对维度 (x_2k, x_2k+1) 共享同一个旋转角，
        # 所以 (Dh/2,) 的频率需要扩展成 (Dh,)。
        cos = theta.cos().repeat_interleave(2, dim=-1)
        sin = theta.sin().repeat_interleave(2, dim=-1)

        # 当 x 是 (B, H, S, Dh) 时，cos/sin 可能还是 (B, S, Dh) 或 (S, Dh)。
        # 只要 cos/sin 的维度数还比 x 少，就继续在 sequence 前面补一个 1 维  
        # 这里持续在 sequence 维前面插入一个维度，使其最终能对 head 维进行广播。
        # 例如：
        # - (S, Dh)    -> (1, S, Dh)    -> (1, 1, S, Dh)
        # - (B, S, Dh) -> (B, 1, S, Dh)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3) # 在倒数第三个位置插一个长度为 1 的新维度
            sin = sin.unsqueeze(-3) # 在倒数第三个位置插一个长度为 1 的新维度

        # 将 cos/sin 的 dtype 对齐到输入，避免 float32 和 bf16/fp16 混合计算带来额外转换。
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        # 4) 对最后一维做偶/奇配对旋转。
        # rotate_tensor(x) 会把：
        # (x_0, x_1, x_2, x_3, ...)
        # 变成：
        # (-x_1, x_0, -x_3, x_2, ...)
        # 然后套用 RoPE 公式：
        # x_rot = x * cos + rotate(x) * sin
        rotated_x = self.rotate_tensor(x)
        return x * cos + rotated_x * sin

    def rotate_tensor(self, x: torch.Tensor) -> torch.Tensor:
        '''
        create a rotated tensor (x_2k, x_2k+1) -> (-x_2k+1, x_2k)
        '''
        # 先把最后一维按两两一组重排：
        # (..., Dh) -> (..., Dh/2, 2)
        # 最后那个长度为 2 的维度分别存放偶数位和奇数位。
        x = rearrange(x, '... (s r) -> ... s r', r=2)

        # 拆出每一对中的偶数位和奇数位。
        x_even, x_odd = x.unbind(dim=-1)

        # 完成二维平面旋转中的“正交向量”构造：
        # (x_even, x_odd) -> (-x_odd, x_even)
        x = torch.stack((-x_odd, x_even), dim=-1)

        # 再还原回原始最后一维的布局，方便和输入逐元素相乘。
        return rearrange(x, '... s r -> ... (s r)')

# 定义 softmax 函数。
# 它会把一组任意实数分数变成“总和为 1 的权重”。
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 下面的文档字符串说明输入张量和归一化维度的含义。
    '''
    x: torch.Tensor Input of the softmax
    dim: int The dimension of x that you want to impelement softmax to.
    '''
    # 先减去当前维度上的最大值。
    # 这一步不会改变 softmax 结果，但能显著提升数值稳定性，
    # 防止 exp 之后出现过大的数。
    x = x - torch.max(x, dim=dim, keepdim=True).values # 第“dim”维度被压成 1

    # 对每个元素取指数，把分数变成正数。
    x = torch.exp(x)

    # 用每个元素除以同一维度上的总和，
    # 得到一个归一化后的概率分布。
    return x / torch.sum(x, dim=dim, keepdim=True)


# 定义缩放点积注意力。
# 这是 Transformer 注意力机制里最核心的一步：
# 先算“谁和谁相关”，再按相关性对 V 做加权求和。
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # 下面的文档字符串给出 q、k、v 和 mask 的常见形状。
    '''
    q: (B, S_q, D)
    k: (B, S_k, D)
    v: (B, S_v, D)
    mask: (B, S_q, S_k) or None
    '''
    # 计算 q 和 k 的点积分数。
    # 这一步在语义上是在问：
    # “第 i 个 query 应该多关注第 j 个 key？”
    # 最终得到的分数张量形状是 (..., S_q, S_k)。
    # 再除以 sqrt(D)，是为了避免维度大时分数过大，softmax 过尖。
    q_k_score = einsum(q, k, '... s_q d, ... s_k d -> ... s_q s_k') / q.size(-1)**0.5 # 得到注意力矩阵
    """
    核心：对于注意力矩阵运算：矩阵大小是 S x Q/K/V  
    S、Q/K/V是最后两个维度。
    D_model 先分别投影成 Q、K、V 然后每个 Q/K/V 再拆成：num_heads * d_k
    但是运算的时候需要 S x Q/K/V  
    所以把注意力头维度直接移到 S 前面，不做数值运算，而是通过广播复制运算操作。

    Q: (B, H, S_q, d_k)
    K: (B, H, S_k, d_k)

    QK^T -> (B, H, S_q, S_k)
    
    其中：
    - d_k 被点积求和掉了
    - B 和 H 保留下来
    - 得到每个 head 的注意力分数矩阵
    """


    # 如果传入了 mask，就把不允许关注的位置直接设成负无穷。
    # 这样它们经过 softmax 后权重会变成 0。
    if mask is not None:
        q_k_score = q_k_score.masked_fill(mask == False, float('-inf'))

    # 在 key 这一维上做 softmax，把原始分数变成注意力权重。
    q_k_attention = softmax(q_k_score, dim=-1)

    # 用注意力权重对 v 做加权求和，
    # 得到每个 query 位置最终聚合出的上下文表示。
    return einsum(q_k_attention, v, '... s_q s_k, ... s_k d -> ... s_q d') # 得到 Z 

class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, position_embedding: nn.Module = RotaryPositionalEmbedding, max_seq_len = None, theta = None, token_positions = None, device=None, dtype=None, use_causal_mask=True):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        use_causal_mask: bool Whether to apply causal masking.
        '''
        super().__init__()
        self.pe = None
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask
        assert d_model % num_heads == 0, 'number of heads donen\' match d_model'
        self.d_k = d_model // num_heads
        self.w_q = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_k = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_v = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_o = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        if position_embedding is not None and max_seq_len is not None and theta is not None:
            self.pe = position_embedding(theta, self.d_k, max_seq_len)
        self.token_positions = token_positions


    def causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0) #在第 0 维插入一个长度为 1 的新维度
        # 插入两次，变成 (1, 1, seq_len, seq_len)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先一次性完成 Q/K/V 的线性投影，仍然保持 (B, S, D_model)。
        q_i = self.w_q(x)
        k_i = self.w_k(x)
        v_i = self.w_v(x)

        # 再把最后一维拆成 (num_heads, d_k)，得到标准多头注意力形状：
        # (B, S, D_model) -> (B, H, S, d_k)
        q_i = rearrange(q_i, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads) 
        k_i = rearrange(k_i, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        v_i = rearrange(v_i, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)


        # RoPE 只作用在 Q/K 上，不作用在 V 上。
        # 因为位置编码的作用是改变注意力分数的相对位置信息，
        # 而 V 负责承载被聚合的内容表示。
        if self.pe is not None:
            q_i = self.pe(q_i, self.token_positions)
            k_i = self.pe(k_i, self.token_positions)
        mask = None
        if self.use_causal_mask:
            # causal mask 形状为 (1, 1, S, S)，会自动广播到 batch 和 head。
            # 这样当前位置只能看到自己以及之前的 token，不能偷看未来信息。
            mask = self.causal_mask(q_i.size(-2)) # q_i.size(-2) 是 q_i 倒数第二维度：S
            mask = mask.to(device=q_i.device)
        atten = scaled_dot_product_attention(q_i, k_i, v_i, mask)

        # 把多头结果重新拼回最后一维，恢复到 (B, S, D_model)。
        atten = rearrange(atten, 'b n_h s d_k -> b s (n_h d_k)', n_h=self.num_heads)
        """
        恢复到 (B, S, D_model)
        恢复到 (B, S, D_model)
        恢复到 (B, S, D_model)
        """

        # 最后再做一次输出投影，得到注意力子层的输出。
        out = self.w_o(atten)
        return out
