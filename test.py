import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from model import LeNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = LeNet().to(device = device)
    module = model.conv1
    print("named_parameters_before:", list(module.named_parameters()))
    print("named_buffers_before:", list(module.named_buffers()))

    # 第一个参数:module，代表要进行剪枝的特定模块，对第一层进行prune
    # 第二个参数:weight说明对weight进行剪枝，而不对bias剪枝
    # 第三个参数:amountkong控制裁减数量，float说明比例，int说明裁剪连接数量
    prune.random_unstructured(module, name="weight", amount=0.3)

    print("named_parameters_after:", list(module.named_parameters()))
    print("named_buffers_after:", list(module.named_buffers()))


    print(module.weight)

    # prune.l1_unstructured(module, name="bias", amount=3)

