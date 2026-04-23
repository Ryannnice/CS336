 # 导入 PyTorch，用于保存和加载检查点。
import torch

 # 导入 torch.nn，方便进行模型类型标注和语义说明。
import torch.nn

 # 定义保存检查点函数。
 # 在这个项目里，save_checkpoint 的调用频率是：
 # 每隔 args.save_intervals 步保存一次（默认值 1000）
def save_checkpoint(model, optimizer, iteration, out):
    # 下面的文档字符串说明这个函数接收哪些对象。
    '''
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    '''
    
    # 使用 torch.save 把当前训练状态序列化到 out。
    torch.save(
        # 保存一个字典，里面包含模型状态、优化器状态和当前迭代步数。
        obj={
            # 保存模型所有参数和 buffer 的状态。
            'model_state': model.state_dict(),

            # 保存优化器内部状态，例如动量、一阶矩、二阶矩等。
            'optimizer_state': optimizer.state_dict(),

            # 保存当前训练到了第几步。
            'iteration': iteration
        },

        # 指定输出位置，可以是路径，也可以是文件对象。
        f=out
    )

 # 定义加载检查点函数。
def load_checkpoint(src, model, optimizer) -> torch.long:
    # 下面的文档字符串说明这个函数会从检查点恢复什么
    '''
    should load a checkpoint from src (path or file- like object), and then recover the model and optimizer states from that checkpoint. 
    Your function should return the iteration number that was saved to the checkpoint. 
    You can usetorch.load(src) to recover what you saved in your save_checkpoint implementation, and the load_state_dict method in both the model and optimizers to return them to their previous states
    该函数应该从 src（路径或类文件对象）加载一个检查点（checkpoint），然后从中恢复模型和优化器的状态。
    你的函数应当返回该检查点中保存的迭代次数（iteration number）。
    你可以使用 torch.load(src) 来恢复你在 save_checkpoint 实现中保存的内容，并调用模型（model）和优化器（optimizer）的 load_state_dict 方法，将它们恢复到之前的状态。

    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    '''

    # 从 src 读取检查点  
    # map_location='cpu' 表示先统一加载到 CPU，避免设备不匹配
    ckp = torch.load(src, map_location='cpu')

    # 用检查点中的模型状态恢复当前模型
    model.load_state_dict(ckp['model_state'])

    # 用检查点中的优化器状态恢复当前优化器
    optimizer.load_state_dict(ckp['optimizer_state'])

    # 返回保存时记录的训练步数
    return ckp['iteration']
