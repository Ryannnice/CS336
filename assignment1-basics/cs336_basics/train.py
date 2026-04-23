#!/usr/bin/env python3
"""
Training script for Transformer language model with wandb and tqdm monitoring.

Example usage:
    # Train with wandb logging
    python train.py --data_dir ./data --wandb_project "my-project" --wandb_run_name "experiment-1"
    
    # Train without wandb
    python train.py --data_dir ./data --no_wandb
"""

# 导入命令行参数解析库，用来读取训练超参数。
import argparse

# 导入 os，用于路径拼接和目录创建。
import os

# 导入 sys；当前文件里没有直接使用，但通常训练脚本会保留它做系统级操作。
import sys

# 导入 json；当前文件里没有直接使用，但常见于配置或日志序列化。
import json

# 导入 PyTorch，用于模型、张量、反向传播和设备管理。
import torch

# 导入 pathlib；当前文件里没有直接使用，但常见于路径处理。
import pathlib

# 导入 NumPy，用于求平均、困惑度等数值统计。
import numpy as np

# 导入 time；当前文件里没有直接使用，但训练脚本里常用于计时。
import time

# 导入 tqdm，用于显示训练进度条。
from tqdm import tqdm

# 导入 wandb，用于实验日志记录。
import wandb

# 导入语言模型主体。
from cs336_basics.model.transformer import transformer_lm

# 导入自定义 AdamW 优化器。
from cs336_basics.trainer.AdamW import AdamW

# 导入 batch 采样函数。
from cs336_basics.trainer.data_loading import data_loading

# 导入训练过程会用到的工具函数：交叉熵、学习率调度、梯度裁剪。
from cs336_basics.trainer.utils import cross_entropy, learning_rate_schedule, gradient_clipping

# 导入 checkpoint 的保存和恢复函数。
from cs336_basics.check_pointing import save_checkpoint, load_checkpoint

# 定义命令行参数解析函数。
def parse_args(): # <--- 这是你自定义的函数名
    # 创建参数解析器。
    parser = argparse.ArgumentParser(description='Train a Transformer language model')
    
    # 定义词表大小参数。
    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of vocabulary')

    # 定义模型隐藏维度参数。
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')

    # 定义 FFN 中间维度参数。
    parser.add_argument('--d_ff', type=int, default=1344, help='FFN dimension')

    # 定义上下文长度参数。
    parser.add_argument('--context_len', type=int, default=256, help='Maximum sequence length')

    # 定义注意力头数参数。
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')

    # 定义 Transformer block 层数参数。
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')

    # 定义 RoPE 的 theta 超参数。
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # 定义最大学习率参数。
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')

    # 定义最小学习率参数。
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate')

    # 定义 warmup 步数参数。
    parser.add_argument('--warm_up_it', type=int, default=500, help='Warmup iterations')

    # 定义余弦退火总步数参数。
    parser.add_argument('--cosine_it', type=int, default=10000, help='Cosine annealing iterations')

    # 定义权重衰减系数参数。
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')

    # 定义 Adam 的 beta1 参数。
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')

    # 定义 Adam 的 beta2 参数。
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')

    # 定义 Adam 的 eps 参数。
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')

    # 定义梯度裁剪阈值参数。
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # 定义 batch size 参数。
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    # 定义总训练步数参数。
    parser.add_argument('--train_steps', type=int, default=6000, help='Total training steps')

    # 定义验证间隔参数。
    parser.add_argument('--val_interval', type=int, default=100, help='Validation interval')

    # 定义每次验证要抽多少个 batch。
    parser.add_argument('--val_batches', type=int, default=10, help='Number of validation batches')

    # 定义 checkpoint 保存间隔参数。
    parser.add_argument('--save_intervals', type=int, default=1000, help='Checkpoint save interval')

    # 定义日志记录间隔参数。
    parser.add_argument('--log_intervals', type=int, default=1, help='Logging interval')

    # 定义 checkpoint 保存目录参数。
    parser.add_argument('--save_ckp_path', type=str, default='./checkpoints', help='Checkpoint save directory')

    # 定义从某个 checkpoint 恢复训练的路径参数。
    parser.add_argument('--resume_ckp', type=str, default=None, help='Path to checkpoint to resume from')
    
    # 定义数据目录参数。
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory path')

    # 定义设备参数。
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, mps')
    
    # 定义 wandb 项目名参数。
    parser.add_argument('--wandb_project', type=str, default='cs336-transformer', help='Wandb project name')

    # 定义 wandb run 名称参数。
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')

    # 定义是否关闭 wandb 的开关参数。
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    # 返回解析好的参数对象。
    return parser.parse_args() # <--- 这是库函数的方法名
    # “自己调用自己”其实是 “命名重合” 导致的视觉错觉：
    # 外层的 parse_args() 是你起的函数名（包装盒）
    # 内层的 parser.parse_args() 是 argparse 库提供的方法名（核心引擎）

# 定义设备选择函数。
def get_device(device_arg):
    """Get the appropriate device"""
    # 如果用户要求自动选择设备，就依次检查 cuda、mps、cpu。
    if device_arg == 'auto':
        # 如果 CUDA 可用，就优先使用 CUDA。
        if torch.cuda.is_available():
            return 'cuda'

        # 否则如果苹果 MPS 可用，就使用 MPS。
        elif torch.backends.mps.is_available():
            return 'mps'

        # 如果都不可用，就退回到 CPU。
        else:
            return 'cpu'

    # 如果用户传的不是 auto，就直接返回用户指定值。
    return device_arg

# 定义用 memmap 方式读取数据集的函数。
"""
np.memmap (Memory-mapped file) 的本质是：把硬盘上的文件，伪装成内存中的数组。
传统读取：硬盘 $\rightarrow$ 内存 $\rightarrow$ CPU 计算。文件多大，内存就得占多大。
memmap 读取：它只在内存里建立一个“索引表”。
只有当真正访问某个索引（比如 dataset[100:200]）时，操作系统才会把这部分数据从硬盘调入内存（Page In）。
"""
def get_dataset_memmap(path, dtype=np.uint16):
    """Load dataset using memory mapping for efficiency"""
    # 如果路径不存在，就直接报错。
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    # 用 NumPy 的 memmap 只映射文件，不一次性全部读入内存。
    dataset = np.memmap(path, dtype=dtype, mode='r')
    # mode='r': 以 只读 (Read-only) 模式打开。这能防止训练脚本意外修改原始数据集，并允许操作系统在多个进程间安全地共享这块映射区域。

    # 返回映射后的数据集对象。
    return dataset

# 定义训练主函数。
def main():
    # 先读取命令行参数。
    args = parse_args()
    
    # 解析并选择训练设备。
    device = get_device(args.device)

    # 打印当前使用的设备。
    print(f"Using device: {device}")
    
    # 如果没有禁用 wandb，就初始化 wandb。
    if not args.no_wandb:
        # 创建一个 wandb run，并把所有参数记录进去。
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

        # 打印 wandb 运行名称。
        print(f"Wandb initialized: {wandb.run.name}")
    
    # 创建 checkpoint 目录；如果目录已存在就不报错。
    os.makedirs(args.save_ckp_path, exist_ok=True)
    
    # 构造 Transformer 语言模型。
    model = transformer_lm(
        vocab_size     =    args.vocab_size,
        context_length =    args.context_len,
        num_layers     =    args.num_layers,
        d_model        =    args.d_model,
        num_heads      =    args.num_heads,
        rope_theta     =    args.rope_theta,
        d_ff           =    args.d_ff
    ).to(device)


    

    # 统计模型总参数量
    total_params = sum(p.numel() for p in model.parameters()) 
    # p.numel: PyTorch 张量（Tensor）的一个方法，返回该张量中包含的元素总数
    # model.parameters(): 是一个生成器（Generator），它会遍历模型中所有被注册为 nn.Parameter 的张量。
    # 内存优势：这里没有使用列表推导式（即没有用 []），这意味着它不会在内存里先创建一个包含所有数字的列表，而是边遍历边累加，非常省内存
    print(f"Model initialized with {total_params} parameters") # 打印参数量。

    # 如果启用了 wandb，就把参数量记到日志里。
    if not args.no_wandb:
        wandb.log({'model/total_parameters': total_params})
    
    # 创建 AdamW 优化器，并把模型参数交给它管理。
    optimizer = AdamW(
        model.parameters(),
        lr           = args.max_lr,
        betas        = (args.beta1, args.beta2),
        eps          = args.eps,
        weight_decay = args.weight_decay
    )
    
    # 如果用户没有给数据目录，就直接报错。
    if args.data_dir is None:
        raise ValueError("Data directory must be specified with --data_dir")
    
    # 拼出训练集 token 文件路径。
    train_data_path = os.path.join(args.data_dir, 'train.dat')
    # 拼出验证集 token 文件路径。
    val_data_path = os.path.join(args.data_dir, 'valid.dat')
    

    # 用 memmap 加载训练数据。
    train_data = get_dataset_memmap(train_data_path)
    # 用 memmap 加载验证数据。
    val_data = get_dataset_memmap(val_data_path)
    

    # 打印训练集 token 数量。
    print(f"Train data size: {len(train_data)} tokens")
    # 打印验证集 token 数量。
    print(f"Val data size: {len(val_data)} tokens")





    # 默认从第 0 步开始训练。
    start_iter = 0

    # 如果用户提供了恢复训练的 checkpoint，就从那里恢复。
    if args.resume_ckp:
        # 打印恢复路径。
        print(f"Resuming from checkpoint: {args.resume_ckp}")

        # 恢复模型、优化器和步数。
        start_iter = load_checkpoint(args.resume_ckp, model, optimizer)

        # 打印恢复后的起始步数。
        print(f"Resumed from iteration {start_iter}")
    




    # 切到训练模式
    model.train()

    # 用列表记录训练过程中每一步的 loss。
    train_losses = []
    
    # 创建训练进度条 progress bar  
    # 本质上是一个 可迭代对象（Iterable）,包裹了你的循环范围 range(...)，每当循环运行一次就内部计数一次，并刷新屏幕上的进度显示
    pbar = tqdm(range(start_iter, args.train_steps), 
                desc    = "Training", 
                
                initial = start_iter, 
                total   = args.train_steps
                ) # default = 6000 
    
    # 遍历每一个训练步。
    # pbar 包裹的是 range(start_iter, args.train_steps)，所以 iter_num 会依次取其中的每一个值
    for iter_num in pbar:
        # 根据当前步数计算学习率。
        lr = learning_rate_schedule(
            iter_num,

            args.max_lr,
            args.min_lr,
            args.warm_up_it,
            args.cosine_it
        )
        
        # 把当前学习率写回优化器的 每个 参数组
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 从训练数据中 随机 采样一个 batch
        input_ids, target_ids = data_loading(
            train_data, # 测试数据
            args.batch_size,
            args.context_len,
            device = device
        )
        
        # 再显式确保输入、目标是 long 类型并放到正确设备上
        input_ids  = input_ids.long().to(device)
        target_ids = target_ids.long().to(device)
        
        # 先清空上一步残留的梯度: 
        optimizer.zero_grad()

        # 做一次前向传播，得到每个位置的词表 logits
        logits = model(input_ids) # model 是 transformer_lm()，接收张量作为输入
        

        # 为了计算交叉熵，把 logits 展平成二维：(batch * seq, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        # 把目标 token id 展平成一维：(batch * seq,)
        targets_flat = target_ids.view(-1) # 这是目标词的 index。会送到cross_entropy()中再查询对应的词
    
        # 计算当前 batch 的语言模型 loss。
        loss = cross_entropy(logits_flat, targets_flat) # in utils.py, line 
        

        # 反向传播，给所有可训练参数计算梯度。
        loss.backward()
        
        # 在更新参数前做梯度裁剪，避免梯度过大。
        gradient_clipping(model.parameters(), args.clip_grad_norm)
        

        # 调用优化器更新参数。
        optimizer.step()

    
        
        # 把当前 loss 记录到历史列表里。
        train_losses.append(loss.item())

        # 如果到了日志记录步，就更新显示并写 wandb。
        if iter_num % args.log_intervals == 0:
            
            # 取最近 100 步的平均 loss；如果还不到 100 步，就对已有 loss 求平均。
            avg_loss = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else np.mean(train_losses)
            # 根据平均 loss 计算 perplexity。
            perplexity = np.exp(avg_loss)
            
            # 把当前统计信息写到进度条后缀里。
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{avg_loss:.4f}',
                'PPL(perplexity)': f'{perplexity:.2f}',
                'LR': f'{lr:.2e}'
            })
            # 如果启用了 wandb，就同步记录训练指标。
            if not args.no_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/avg_loss': avg_loss,
                    'train/perplexity': perplexity,
                    'train/learning_rate': lr,
                    'iteration': iter_num
                })
        



        # 验证步：
        if iter_num % args.val_interval == 0 and iter_num > 0:
            # 切到评估模式。
            model.eval()

            # 用列表收集验证损失。
            val_losses = []
            
            # 关闭梯度计算，节省显存和计算。
            with torch.no_grad():
                # 连续采样若干个验证 batch。
                for _ in range(args.val_batches): # 采集“val_batches”条数据：
                    # 从验证数据中采样 batch。
                    # 注意：这里的调用方式和当前 data_loading 的签名不完全一致，
                    # 阅读时先把它理解成“验证时也需要取 input/target batch”。
                    val_input_ids, val_target_ids = data_loading(
                        val_data, # 验证数据
                        args.batch_size,
                        args.context_len,
                        device = device
                    )
                    
                    """
                    # 把验证输入转成 long 并移动到设备上。
                    val_input_ids = torch.from_numpy(val_input_ids).long().to(device)
                    # 把验证目标转成 long 并移动到设备上。
                    val_target_ids = torch.from_numpy(val_target_ids).long().to(device)
                    """
                    # data_loading 已经返回 tensor，这里只需确保类型和设备正确。
                    val_input_ids = val_input_ids.long().to(device)
                    # 目标张量同理处理。
                    val_target_ids = val_target_ids.long().to(device)
                    

                    # 前向传播得到验证 logits。
                    val_logits = model(val_input_ids)

                    # 把验证 logits 展平成二维。
                    val_logits_flat = val_logits.view(-1, val_logits.size(-1))

                    # 把验证目标展平成一维。
                    val_targets_flat = val_target_ids.view(-1)
                    
                    # 计算当前验证 batch 的 loss。
                    val_loss = cross_entropy(val_logits_flat, val_targets_flat)

                    # 记录当前验证 loss。
                    val_losses.append(val_loss.item())
            
            # 对所有验证 batch 的 loss 求平均。
            avg_val_loss = np.mean(val_losses)

            # 根据验证平均 loss 计算验证 perplexity。
            val_perplexity = np.exp(avg_val_loss)
            
            # 在终端打印验证指标。
            tqdm.write(f"Validation | Loss: {avg_val_loss:.4f} | PPL: {val_perplexity:.2f}")
            
            # 如果启用了 wandb，就记录验证指标。
            if not args.no_wandb:
                wandb.log({
                    'val/loss': avg_val_loss,
                    'val/perplexity': val_perplexity,
                    'iteration': iter_num
                })
            
            # 验证结束后切回训练模式。
            model.train()
        



        # 如果到了 checkpoint 保存步，就保存一次训练现场。
        if iter_num % args.save_intervals == 0 and iter_num > 0:
            # 生成当前 checkpoint 文件名。
            checkpoint_path = os.path.join(args.save_ckp_path, f'checkpoint_{iter_num}.pt')

            # 保存模型、优化器和当前步数。
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)

            # 在终端打印保存成功信息。
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")
    
    # 训练结束后关闭进度条。
    pbar.close()
    

    # 生成最终 checkpoint 的文件路径。
    final_checkpoint_path = os.path.join(args.save_ckp_path, f'checkpoint_final_{args.train_steps}.pt')
    # 保存训练结束时的最终 checkpoint。
    save_checkpoint(model, optimizer, args.train_steps, final_checkpoint_path)
    # 打印最终 checkpoint 路径。
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    # 打印训练完成信息。
    print("Training completed!")

    # 如果启用了 wandb，就正常结束本次 run。
    if not args.no_wandb:
        wandb.finish()



# 如果当前文件是作为主程序运行，就执行 main。
if __name__ == '__main__':
    main()

"""
  一个 Python 文件有两种用法：  
  1. 直接运行: python train.py  
  2. 被别的文件 import: import train  

  - 直接运行时, Python 会把这个文件的 __name__ 设成 '__main__'  
  - 被 import 时, __name__ 会变成模块名，比如 'train'  
"""
