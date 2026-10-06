"""
三种初始化方法对比实验
"""
import torch
import torch.nn as nn
import math


def default_init(model):
    """
    PyTorch默认初始化（作为baseline）
    Linear层默认使用 kaiming_uniform_ (a=sqrt(5))
    """
    # PyTorch默认已经初始化好了，这里什么都不做
    print("使用PyTorch默认初始化")
    pass


def xavier_init(model):
    """
    Xavier初始化（适用于Sigmoid/Tanh）
    公式: 权重方差 = 2 / (n_in + n_out)

    TODO: 请实现Xavier初始化
    提示:
    1. 遍历model的所有层: for layer in model.get_all_layers()
    2. 对每一层的权重 layer.weight 应用初始化
    3. PyTorch提供了 nn.init.xavier_uniform_(tensor) 函数
    4. 偏置 layer.bias 初始化为0: nn.init.zeros_(layer.bias)
    """
    print("使用Xavier初始化")
    # ========== 你的代码开始 ==========
    for layer in model.get_all_layers():
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    # ========== 你的代码结束 ==========


def kaiming_init(model):
    """
    Kaiming初始化（适用于ReLU）
    公式: 权重方差 = 2 / n_in

    TODO: 请实现Kaiming初始化
    提示:
    1. 遍历model的所有层: for layer in model.get_all_layers()
    2. 对每一层的权重 layer.weight 应用初始化
    3. PyTorch提供了 nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu') 函数
       - mode='fan_in' 表示只考虑输入神经元数量 n_in
       - nonlinearity='relu' 表示针对ReLU设计
    4. 偏置 layer.bias 初始化为0: nn.init.zeros_(layer.bias)
    """
    print("使用Kaiming初始化")
    # ========== 你的代码开始 ==========
    for layer in model.get_all_layers():
        torch.nn.init.kaiming_uniform_(tensor=layer.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(layer.bias)

    # ========== 你的代码结束 ==========


# 初始化方法字典（方便训练脚本调用）
INIT_METHODS = {
    'default': default_init,
    'xavier': xavier_init,
    'kaiming': kaiming_init
}
