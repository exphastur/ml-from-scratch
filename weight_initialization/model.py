"""
5层全连接网络，用于测试不同初始化方法
"""
import torch
import torch.nn as nn


class FiveLayerNet(nn.Module):
    """
    5层全连接网络
    结构: 784 -> 512 -> 256 -> 128 -> 64 -> 10
    """
    def __init__(self):
        super(FiveLayerNet, self).__init__()

        # 定义5层全连接
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

        # 使用ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入: (batch_size, 28, 28)
        x = x.view(-1, 784)  # 展平成 (batch_size, 784)

        # 5层前向传播
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # 最后一层不用激活函数（交叉熵会处理）

        return x

    def get_all_layers(self):
        """返回所有全连接层，用于初始化"""
        return [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
