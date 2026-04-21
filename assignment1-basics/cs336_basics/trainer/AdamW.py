 # 导入 PyTorch，用于张量计算和参数更新。
import torch

 # 导入 nn；虽然本文件里没有直接用到 nn.Module，
 # 但优化器处理的对象通常是神经网络参数。
from torch import nn

 # 导入 PyTorch 优化器基类，当前自定义 AdamW 会继承它。
from torch.optim import Optimizer

 # 导入类型标注用到的 Callable 和 Iterable。
from collections.abc import Callable, Iterable

 # 导入 Optional，便于给可选参数做类型说明。
from typing import Optional

 # 导入 math，用于偏差修正里的平方根计算。
import math

 # 定义自定义 AdamW 优化器。
class AdamW(Optimizer):
    # 初始化优化器。
    def __init__(self,params: Iterable[torch.nn.parameter.Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.01):
        # if alpha<0:
        #     raise ValueError(f'invalid value input for alpha: {alpha}')
        # if beta[0]<0:
        #     raise ValueError(f'invalid input for beta_1: {beta[0]}')
        # if beta[1]<0:
        #     raise ValueError(f'invalid input for beta_2: {beta[1]}')
        # if eps<0:
        #     raise ValueError(f'invalid input for eps: {eps}')
        # if lamb<0:
        #     raise ValueError(f'invalid input for lambda: {lamb}')
        
        # 把超参数整理成默认配置字典。
        # 这些值会被 Optimizer 基类保存到 param_groups 中。
        defaults = {
            # 学习率。
            'alpha': lr,

            # 一阶矩动量系数 beta1。
            'beta1': betas[0],

            # 二阶矩动量系数 beta2。
            'beta2': betas[1],

            # 数值稳定项，防止分母为 0。
            'eps': eps,

            # 权重衰减系数。
            'lamb': weight_decay
        }

        # 调用父类构造函数，完成参数分组和默认超参数注册。
        super().__init__(params, defaults)




    # 执行一次参数更新：
    def step(self, closure: Callable | None = None): # core
        """
        closure: Callable | None = None:
        Callable: 这是一个类型提示(Type Hint)，表示这个参数必须是一个“可调用对象”(通常是一个函数)。
        | None: 表示这个参数也可以是 None(空)。
        = None: 默认值为 None。这意味着在大多数情况下(如简单的 SGD)，你可以直接调用 step() 而不传任何参数。
        """
        # 先准备一个 loss 变量，和 PyTorch 优化器接口保持一致。
        loss = None

        # 如果传入了 closure，就先执行它。
        # closure 一般用于某些需要重复前向的优化器，这里主要是接口兼容。
        if closure is not None:
            closure()

        # 下面这段注释解释 param_groups 的结构：
        # 优化器内部会把参数按组保存，每组都有自己的超参数配置。
        '''
        param_groups looks like a bunch of (learnable para, hypter_para) pairs:
        [
        {"params": [...layer1 参数...], "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        {"params": [...layer2 参数...], "lr": 5e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
        ]
        '''

        # 遍历每一组参数。
        for group in self.param_groups:

            # 再遍历这组里的每个参数张量。
            for p in group['params']:

                # 如果当前参数没有梯度，就跳过。
                if p.grad is None:
                    continue

                # 取出当前参数的梯度。
                grad = p.grad.data

                # 如果梯度是稀疏张量，就报错。
                # 当前这个自定义 AdamW 不支持稀疏梯度。
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                # 取出当前参数组的超参数。
                alpha = group['alpha']
                beta_1 = group['beta1']
                beta_2 = group['beta2']
                eps = group['eps']
                lamba = group['lamb']

                # 取出当前参数对应的状态字典。
                # 这里面会保存 m、v、t 等历史统计量。
                state = self.state[p]

                # 读取上一时刻的一阶矩 m。
                # 如果这是第一次更新，就用全 0 张量初始化。
                prev_m = state.get('m', torch.zeros_like(grad))

                # 更新一阶矩：
                # m_t = beta1 * m_{t-1} + (1-beta1) * grad
                state['m'] = beta_1*prev_m+(1-beta_1)*grad

                # 读取上一时刻的二阶矩 v。
                prev_v = state.get('v', torch.zeros_like(grad))

                # 更新二阶矩：
                # v_t = beta2 * v_{t-1} + (1-beta2) * grad^2
                state['v'] = beta_2*prev_v+(1-beta_2)*torch.square(grad)

                # 读取当前时间步 t。
                # 如果是第一次更新，就从 1 开始。
                t = state.get('t', 1)

                # 做偏差修正，得到当前时刻有效学习率 alpha_t。
                alpha_t = alpha*math.sqrt(1-beta_2**t)/(1-beta_1**t)
                # beta < 1  =>  t 越小，beta**t 越大，1 - beta 越小
                # 在训练初期（$t$ 较小时），一阶和二阶矩还没攒够“经验”，这个系数会把步长拉回正常水平，防止模型起步时乱跑。

                # 执行 Adam 主更新：
                # p = p - alpha_t * m / (sqrt(v) + eps)
                p.data -= alpha_t*state['m']/(torch.sqrt(state['v'])+eps)
                # 如果要循环几百万次去算每一个元素，那模型训练岂不是慢死了？
                # 实际上，底层并没有写 for 循环。
                # GPU（或者 CPU 的向量指令集）使用的是 SIMD
                # Single Instruction, Multiple Data，单指令多数据流

                # 执行权重衰减：
                # p = p - alpha * lambda * p
                p.data -= alpha*lamba*p.data

                # 时间步加 1，供下一次更新使用。
                state['t'] = t+1

        # 返回 loss，与 PyTorch 优化器接口保持一致。
        return loss
