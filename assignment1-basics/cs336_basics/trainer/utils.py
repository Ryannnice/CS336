 # 导入 PyTorch，用于张量运算和自动求导。
import torch

 # 导入 torch.nn；虽然这个文件里当前没有直接使用 nn，
 # 但它通常会和训练工具函数放在一起。
from torch import nn

 # 导入 math，用来计算余弦学习率调度中的 cos 和 pi。
import math

 # 导入 Iterable，用于给参数集合做类型标注。
from collections.abc import Iterable

 # 定义交叉熵函数：输入是 logits 和正确答案 token id，输出是平均损失。
def cross_entropy(out_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # 下面的文档字符串解释这个函数的输入和输出语义。
    '''
    out_logit: torch.Tensor Output of the Linear of the transformer. Shape is like (batch_size, vocab_size)
    target: torch.Tensor We are using next-word prediction, so the target is the next word of our input. Shape is like (batch_size)
    '''
    # 取出每个样本里“正确类别”对应的那个 logit。
    # target.unsqueeze(-1) 会把形状从 (B,) 变成 (B, 1)，
    # 这样 gather 才能在最后一维按索引抓取。
    get_logit = out_logit.gather(dim=-1, index=target.unsqueeze(-1))

    # 对每个样本在 vocab 维度上做 logsumexp。
    # 这是 softmax 分母取对数后的稳定写法。
    logsumexp = torch.logsumexp(input=out_logit, dim=-1, keepdim=True)

    # 交叉熵可以写成：logsumexp(logits) - 正确类别的 logit。
    # 这里得到的是每个样本各自的 loss，形状是 (B, 1)。
    loss = -get_logit + logsumexp

    # 对 batch 维做平均，得到一个标量损失。
    return torch.mean(loss,dim=0, keepdim=False)
    

 # 定义学习率调度函数：前期线性 warmup，中期余弦衰减，后期保持最小学习率。
def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # 保证余弦退火的总迭代数大于 warmup 阶段，否则调度区间无效。
    assert cosine_cycle_iters>warmup_iters, 'Invalid input for iteration striction'

    # 如果当前还在 warmup 阶段，就从 0 线性增加到最大学习率。
    if it<warmup_iters:
        return it*max_learning_rate/warmup_iters

    # 如果当前处在余弦退火阶段，就按照 cosine 曲线从 max_lr 衰减到 min_lr。
    elif warmup_iters<=it<=cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)*math.pi/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)

    # 如果已经超过退火阶段，就保持最小学习率不再变化。
    else:
        return min_learning_rate


 # 定义梯度裁剪函数，防止梯度过大导致训练不稳定。
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 设定一个很小的常数，避免后面除以 0。
    eps = 1e-6

    # 只收集那些确实有梯度的参数。
    grads = [p.grad for p in parameters if p.grad is not None]

    # 初始化梯度总范数。
    L2_norm = 0.0

    # 逐个梯度累加平方和。
    for g in grads:
        L2_norm += (g.data**2).sum()

    # 最后开平方，得到整体的 L2 范数。
    L2_norm = torch.sqrt(L2_norm)

    # 如果总范数本来就小于阈值，就什么都不做。
    if L2_norm < max_l2_norm:
        pass

    # 如果总范数超过阈值，就按比例把每个梯度一起缩小。
    else:
        for g in grads:
            g.data *= max_l2_norm/(L2_norm+eps)
    
