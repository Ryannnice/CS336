#!/usr/bin/env python3
"""
Transformer 语言模型的文本生成脚本。

这个脚本实现了解码流程，包括：
- temperature 缩放
- top-p（nucleus）采样
- 使用训练好的模型逐 token 生成文本
"""

# 导入命令行参数解析库，用来读取生成时的各种参数。
import argparse

# 导入 PyTorch，用于张量、模型前向和采样。
import torch

# 导入 NumPy；当前文件里没有直接使用，但保留原始依赖声明
import numpy as np

# 导入类型标注工具，帮助说明函数参数和返回值类型
from typing import List, Optional, Union

# 导入 sys，用来修改 Python 的模块搜索路径
import sys

# 导入 os，用来处理文件路径
import os

# 把项目根目录加入 Python 路径，确保当前脚本能导入仓库里的模块。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 Transformer 语言模型主体。
from cs336_basics.model.transformer import transformer_lm

# 导入 checkpoint 加载函数；当前文件主体逻辑里没有直接调用，但保留原始导入。
from cs336_basics.check_pointing import load_checkpoint

# 导入 AdamW；当前文件主体逻辑里没有直接使用，但保留原始导入。
from cs336_basics.trainer.AdamW import AdamW


# 定义带 temperature 的 softmax 函数
def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    对 logits 先做 temperature 缩放，再做 softmax, 得到概率分布

    Args:
        logits: 原始 logits, 形状通常是 (..., vocab_size)
        temperature: 温度参数；越小分布越尖锐，越大分布越平缓

    Returns:  
        加入 temperature 后的 softmax 概率  
    """  
    # 先用 temperature 缩放 logits  
    scaled_logits = logits / temperature  

    # 为了数值稳定，先找出最后一维的最大值  
    max_logits = torch.max(scaled_logits, dim=-1, keepdim=True)[0] # [0]: value; [1]: index  

    # 每个 logit 先减去最大值，再做指数运算，避免数值过大  
    exp_logits = torch.exp(scaled_logits - max_logits)  

    # 用指数值除以总和，得到规范化后的概率分布  
    probabilities = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)  

    # 返回概率分布    
    return probabilities # 是个矩阵    


# 定义 top-p 采样前的概率裁剪函数
def top_p_sampling(probabilities: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    对概率分布应用 top-p (nucleus)采样, 只保留累计概率达到阈值 p 的那部分 token。

    Args: 
        probabilities: 输入概率分布，形状通常是 (..., vocab_size)
        p: 累计概率阈值

    Returns:
        过滤并重新归一化后的概率分布
    """
    # 按最后一维从大到小[排序]，方便做累计概率截断
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    # sorted_indices: 已排序索引 

    # 沿最后一维做累计求和，得到累计概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    """
    cumsum = cumulative sum = 累计和

    比如： x = [0.4, 0.3, 0.2, 0.1]
    那么： torch.cumsum(x, dim=-1)
    结果是： [0.4, 0.7, 0.9, 1.0]
    """

    # 标记累计概率还没有超过 p 的那些 token。
    mask = cumulative_probs <= p # 如果没达到，mask 标记为 1
    """
    top-p: 累积概率表  
    因为 top-p 的目标不是： “每个 token 概率必须大于某个阈值”  
    而是： “保留一小撮最可能的 token, 使它们加起来至少覆盖总概率的 p”  

    top-p 的设计思想是：  
    - 模型很确定时，候选集合自动变小  
    - 模型不确定时，候选集合自动变大  
    """

    # 无论如何都保留当前概率最大的 token，避免全部被过滤掉   
    mask[..., 0] = True # 取最后一维上索引为 0 的那一列 / 那个位置（排序后，这是最大值）  
    # 对不在 nucleus 里的 token 置零  
    filtered_probs = sorted_probs * mask.float()   
    # 对剩余概率重新归一化，使最后一维和重新变成 1： 
    filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)   
    """
    filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True): 

     [1][2][3][4][5]   /   [5]         [ ][ ][ ][ ][ ]    
     [1][2][3][4][5]   /   [5]    =    [ ][ ][ ][ ][ ]    
     [5][6][7][8][9]   /   [9]         [ ][ ][ ][ ][ ]    
     [5][6][7][8][9]   /   [9]         [ ][ ][ ][ ][ ]    
    
    """

    # 创建一个[和原始输入同形状]的全零张量，用来放回原顺序下的概率   
    output_probs = torch.zeros_like(probabilities)   
    
    # 按照排序前的原始索引，把过滤后的概率 scatter 回去  
    output_probs.scatter_(-1, sorted_indices, filtered_probs)  
    """
    Index: 是原始位置的索引  
    Sorted_index: “原索引12345”的顺序打乱, 但是对应的value绑定不变  
    所以index还是指向排序原来的位置, 可以索引回去  
    """
    """  
    如果形状是：(B, S, V), 那么可以理解成：  
    - output_probs[b, s, :]: 是第 b 个样本、第 s 个位置的原词表顺序概率向量  
    - scatter_(-1, ...): 就是对每个 (b, s) 单独处理这条长度为 V 的向量  

    “对于每个位置，都放回各自操作后的概率值” 

    
    -1: 表示沿最后一维写
    当前是一维向量时，最后一维就是这唯一的一维，所以等价于: scatter_(0, ...) 

    如果形状是：(B, S, V), 那么可以理解成：
    - output_probs[b, s, :]  是第 b 个样本、第 s 个位置的原词表顺序概率向量
    - scatter_(-1, ...)  就是对每个 (b, s) 单独处理这条长度为 V 的向量
    """

    # 返回恢复到原词表顺序的概率分布。
    return output_probs


# 定义核心文本生成函数。
def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
    eos_token: Optional[str] = "<|endoftext|>"
) -> str:
    """
    用训练好的语言模型从 prompt 出发逐 token 生成文本。

    Args:
        model: 已训练好的 Transformer 语言模型
        tokenizer: 负责文本和 token id 相互转换的 tokenizer
        prompt: 生成起点文本
        max_tokens: 最多新生成多少个 token
        temperature: 采样温度； 越小logits输出差距越大, 越保守； 越大logits输出差距越小, 越随机
        top_p: nucleus sampling 的累计概率阈值
        device: 生成所使用的设备
        eos_token: 终止生成的特殊 token

    Returns:
        最终生成出的完整文本
    """
    # 切到评估模式，告诉模型接下来是推理而不是训练
    model.eval()

    # 先把 prompt 文本编码成 token id 列表
    prompt_tokens = tokenizer.encode(prompt)

    # 把 token 列表转成形状为 (1, seq_len) 的张量，并移动到指定设备
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # 复制一份生成结果列表，初始内容就是 prompt 本身对应的 token
    generated_tokens = prompt_tokens.copy()

    # 生成阶段不需要计算梯度，所以关闭 autograd。
    with torch.no_grad():
        # 最多循环生成 max_tokens 个新 token。
        for _ in range(max_tokens):

            # 把当前输入序列送入模型，得到每个位置的 logits。
            logits = model(input_ids)

            
            # 只取最后一个位置的 logits，因为我们现在只关心“下一个 token”。
            next_token_logits = logits[0, -1, :] 
            # 把最后一个位置的 logits 变成概率分布。
            probabilities = softmax_with_temperature(next_token_logits, temperature)
            # 如果要求 top-p 采样，就进一步裁剪概率分布。
            if top_p < 1.0:
                probabilities = top_p_sampling(probabilities, top_p)
            # 按概率分布随机采样一个下一个 token。
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            # 把新采样到的 token id 追加到结果列表里。
            generated_tokens.append(next_token)

            # 如果指定了终止 token，就检查当前生成是否应该停止。
            if eos_token is not None:
                # 把当前新 token 单独解码成字符串。
                decoded_token = tokenizer.decode([next_token])

                # 如果它就是结束符，就提前停止生成。
                if decoded_token.strip() == eos_token.strip():
                    break
                

            # 把刚生成的 token 包装成形状为 (1, 1) 的张量。
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            # 把这个 token 拼接到输入序列末尾，供下一轮继续预测。
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)


            # 如果当前序列长度超过模型上下文窗口，就只保留最后 context_length 个 token。
            if input_ids.size(1) > model.context_length:
                input_ids = input_ids[:, -model.context_length:]


    # 把所有生成得到的 token id 解码回文本。
    generated_text = tokenizer.decode(generated_tokens)

    # 返回最终文本。
    return generated_text


# 定义加载模型和 tokenizer 的辅助函数。
def load_model_and_tokenizer(checkpoint_path: str, vocab_path: str, merges_path: str, device: str = "cpu"):
    """
    从 checkpoint 和 tokenizer 文件中恢复模型与 tokenizer。

    Args:
        checkpoint_path: 模型 checkpoint 路径
        vocab_path: tokenizer 的 vocab 文件路径
        merges_path: tokenizer 的 merges 文件路径
        device: 模型要加载到哪个设备

    Returns:
        (model, tokenizer) 二元组
    """
    # 尝试导入 tokenizer 类。
    try:
        # 从项目里的 tokenizer 模块导入 Tokenizer。
        from cs336_basics.tokenizer import Tokenizer

        # 根据 vocab 和 merges 文件恢复 tokenizer。
        tokenizer = Tokenizer.from_files(vocab_path, merges_path)

    # 如果导入失败，就进入兜底逻辑。
    except ImportError:
        # 打印告警信息，提醒 tokenizer 导入有问题。
        print("Warning: Could not import tokenizer. You may need to implement or adjust import path.")

        # 将 tokenizer 设为 None，供后续主流程判断。
        tokenizer = None

    # 先把 checkpoint 文件加载到指定设备上。
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 如果 checkpoint 里显式存了模型配置，就直接读取。
    if 'model_config' in checkpoint:
        # 从 checkpoint 中拿到模型配置字典。
        config = checkpoint['model_config']

    # 否则就退回到默认配置。
    else:
        # 构造一个默认配置字典；这要求 checkpoint 权重和默认结构兼容。
        config = {
            'vocab_size': 10000,
            'context_length': 256,
            'num_layers': 4,
            'd_model': 512,
            'num_heads': 16,
            'rope_theta': 10000.0,
            'd_ff': 1344
        }

        # 打印提醒，说明当前是在用默认模型配置。
        print("Warning: Using default model configuration. Adjust as needed.")

    # 根据配置创建 Transformer 语言模型实例，并移动到指定设备。
    model = transformer_lm(
        vocab_size=config.get('vocab_size', 10000),
        context_length=config.get('context_length', 256),
        num_layers=config.get('num_layers', 4),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 16),
        rope_theta=config.get('rope_theta', 10000.0),
        d_ff=config.get('d_ff', 1344)
    ).to(device)

    # 如果 checkpoint 里是标准保存格式，就取其中的 model_state_dict 来加载。
    if 'model_state_dict' in checkpoint:
        # 把模型权重加载进 model。
        model.load_state_dict(checkpoint['model_state_dict'])

    # 否则就把整个 checkpoint 当成 state_dict 来加载。
    else:
        # 直接加载 checkpoint 本体。
        model.load_state_dict(checkpoint)

    # 返回恢复好的模型和 tokenizer。
    return model, tokenizer


# 定义命令行主函数。
def main():
    # 创建参数解析器，用来读取命令行输入。
    parser = argparse.ArgumentParser(description='Generate text with a trained Transformer language model')

    # 添加 checkpoint 路径参数。
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')

    # 添加 tokenizer vocab 文件路径参数。
    parser.add_argument('--vocab', type=str, required=True, help='Path to tokenizer vocabulary file')

    # 添加 tokenizer merges 文件路径参数。
    parser.add_argument('--merges', type=str, required=True, help='Path to tokenizer merges file')

    # 添加 prompt 参数，指定生成起始文本。
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Input prompt for generation')

    # 添加最大生成 token 数参数。
    parser.add_argument('--max_tokens', type=int, default=256, help='Maximum number of tokens to generate')

    # 添加 temperature 参数。
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling (lower = more deterministic)')

    # 添加 top-p 参数。
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p threshold for nucleus sampling')

    # 添加一次要生成多少个样本的参数。
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')

    # 添加设备参数。
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, mps')

    # 添加结束符参数。
    parser.add_argument('--eos_token', type=str, default='<|endoftext|>', help='End-of-sequence token')

    # 解析命令行参数。
    args = parser.parse_args()

    # 如果用户要求自动选设备，就按 CUDA、MPS、CPU 的顺序判断。
    if args.device == 'auto':
        # 如果 CUDA 可用，就优先使用 CUDA。
        if torch.cuda.is_available():
            device = 'cuda'

        # 否则如果苹果 MPS 可用，就使用 MPS。
        elif torch.backends.mps.is_available():
            device = 'mps'

        # 如果都不可用，就退回到 CPU。
        else:
            device = 'cpu'

    # 如果用户明确指定了设备，就直接采用。
    else:
        device = args.device

    # 打印当前实际使用的设备。
    print(f"Using device: {device}")

    # 提示即将开始加载模型和 tokenizer。
    print("Loading model and tokenizer...")

    # 尝试加载模型和 tokenizer。
    try:
        # 调用辅助函数恢复模型与 tokenizer。
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            merges_path=args.merges,
            device=device
        )

        # 如果成功，就打印成功提示。
        print("Model and tokenizer loaded successfully!")

    # 如果加载过程中报错，就打印错误并退出。
    except Exception as e:
        # 把异常信息打印出来，方便排查。
        print(f"Error loading model or tokenizer: {e}")

        # 提前结束 main。
        return

    # 如果 tokenizer 还是 None，说明前面导入或加载失败。
    if tokenizer is None:
        # 打印错误提示。
        print("Error: Tokenizer could not be loaded. Please check your tokenizer implementation.")

        # 提前结束 main。
        return

    # 统计并打印模型总参数量。
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # 打印本次生成任务的基本信息。
    print(f"\nGenerating {args.num_samples} sample(s) with prompt: '{args.prompt}'")

    # 打印生成超参数。
    print(f"Parameters: max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p}")

    # 打印分隔线，方便阅读输出。
    print("-" * 80)

    # 按要求生成多个样本。
    for i in range(args.num_samples):
        # 如果一共要生成多个样本，就打印样本编号。
        if args.num_samples > 1:
            # 打印当前样本序号。
            print(f"\nSample {i+1}:")

            # 打印样本内部分隔线。
            print("-" * 40)

        # 尝试执行单次文本生成。
        try:
            # 调用 generate_text 生成完整文本。
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                eos_token=args.eos_token
            )

            # 打印生成结果文本。
            print(generated_text)

            # 打印样本结束分隔线。
            print("\n" + "=" * 80)

        # 如果生成过程中出错，就打印错误并停止后续样本生成。
        except Exception as e:
            # 输出异常信息。
            print(f"Error during generation: {e}")

            # 跳出循环。
            break


# 如果当前文件是直接运行的主程序，就进入 main。
if __name__ == '__main__':
    # 执行命令行主函数。
    main()
