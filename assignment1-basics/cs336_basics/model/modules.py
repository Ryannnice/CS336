 # 导入 PyTorch 主包，后面会用它创建张量和参数。
import torch 

 # 导入 PyTorch 的神经网络模块基类 nn.Module，以及参数容器等工具。
import torch.nn as nn 

 # 导入 einops 里的重排与张量乘法工具，后面注意力和 RoPE 会频繁用到。
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
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

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
        std = (2 / (self.in_features + self.out_features)) ** 0.5

        # 用截断正态分布初始化权重。
        # 权重不会初始化得过大，有助于训练稳定。
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


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g_weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self._init_weight()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        out = einsum(x/rms, self.g_weight, '... d, d -> ... d')
        return out.to(dtype=in_dtype)
    
    def _init_weight(self):
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
        # 这里持续在 sequence 维前面插入一个维度，使其最终能对 head 维进行广播。
        # 例如：
        # - (S, Dh)    -> (1, S, Dh)    -> (1, 1, S, Dh)
        # - (B, S, Dh) -> (B, 1, S, Dh)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

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


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    x: torch.Tensor Input of the softmax
    dim: int The dimension of x that you want to impelement softmax to.
    '''
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)



def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
    q: (B, S_q, D)
    k: (B, S_k, D)
    v: (B, S_v, D)
    mask: (B, S_q, S_k) or None
    '''
    q_k_score = einsum(q, k, '... s_q d, ... s_k d -> ... s_q s_k') / q.size(-1)**0.5
    #add mask
    if mask is not None:
        q_k_score = q_k_score.masked_fill(mask == False, float('-inf'))
    q_k_attention = softmax(q_k_score, dim=-1)
    return einsum(q_k_attention, v, '... s_q s_k, ... s_k d -> ... s_q d')

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
        return mask.unsqueeze(0).unsqueeze(0)


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
            mask = self.causal_mask(q_i.size(-2))
            mask = mask.to(device=q_i.device)
        atten = scaled_dot_product_attention(q_i, k_i, v_i, mask)

        # 把多头结果重新拼回最后一维，恢复到 (B, S, D_model)。
        atten = rearrange(atten, 'b n_h s d_k -> b s (n_h d_k)', n_h=self.num_heads)

        # 最后再做一次输出投影，得到注意力子层的输出。
        out = self.w_o(atten)
        return out
