 # 导入 PyTorch，用来创建张量并把数据放到指定设备上。
import torch

 # 导入 NumPy 的类型标注，方便说明 dataset 的类型。
import numpy.typing as npt

 # 定义一个数据采样函数：从一维 token 数据集中随机切出若干段输入和目标。
def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 下面的文档字符串说明这个函数的作用和输入输出。
    """
    Sample a batch of inputs and targets from the dataset.

    Args:
        dataset (npt.NDArray): The dataset to sample from.
        batch_size (int): The number of samples in the batch.
        context_length (int): The length of each input sequence.
        device (torch.device | None): The device to place tensors on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the input sequences and their corresponding targets.
    """
    # 记录整个数据集的长度，后面随机采样起点时要用到。
    dataset_len = len(dataset)

    # 预先创建输入张量，形状是 (batch_size, context_length)，元素类型是 long。
    inputs = torch.empty(batch_size, context_length, dtype=torch.long)

    # 预先创建目标张量，形状和 inputs 一样，也是 long 类型。
    targets = torch.empty(batch_size, context_length, dtype=torch.long)

    # 用一个循环逐条构造 batch 中的样本。
    for i in range(batch_size):
        # 随机选择当前样本的起始位置。
        # 这里要保证后面还能取到 context_length 个输入 token，
        # 以及再往后 1 位的目标 token。
        start_idx = torch.randint(0, dataset_len-context_length, (1, )).item()

        # 从数据集中切出一段连续 token，作为模型输入。
        input_seq = dataset[start_idx: start_idx+context_length]

        # 再把这段序列整体向右平移一位，作为“下一个 token”的监督信号。
        input_target = dataset[start_idx+1: start_idx+context_length+1]

        # 把当前输入序列转成 PyTorch 张量，并放进第 i 条样本。
        inputs[i] = torch.tensor(input_seq, dtype=torch.long)

        # 把当前目标序列转成 PyTorch 张量，并放进第 i 条样本。
        targets[i] = torch.tensor(input_target, dtype=torch.long)
    
    # 把输入张量移动到指定设备，例如 cpu 或 cuda。
    inputs = inputs.to(device=device)

    # 把目标张量也移动到同一个设备上。
    targets = targets.to(device=device)

    # 返回一对张量：第一个是模型输入，第二个是监督目标。
    return (inputs, targets)


'''
最后得到两个张量：
  - inputs.shape = (batch_size, context_length)
  - targets.shape = (batch_size, context_length)
'''