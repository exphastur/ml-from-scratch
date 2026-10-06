# 权重初始化实验：Xavier vs Kaiming

## 实验目的

对比三种权重初始化方法在MNIST手写数字分类任务上的表现，验证初始化理论。

## 实验设计

**网络结构：**
- 5层全连接网络：784 → 512 → 256 → 128 → 64 → 10
- 激活函数：ReLU
- 损失函数：交叉熵
- 优化器：SGD (lr=0.01, momentum=0.9)
- 训练轮数：10 epochs
- 批大小：128

**三种初始化方法：**
1. **Default**：PyTorch默认初始化（kaiming_uniform with a=sqrt(5)）
2. **Xavier**：`权重方差 = 2/(n_in + n_out)`，适用于Sigmoid/Tanh
3. **Kaiming**：`权重方差 = 2/n_in`，适用于ReLU

## 实验结果

| 初始化方法 | 第1轮测试准确率 | 最终测试准确率（第10轮）| 训练时间 |
|-----------|----------------|---------------------|---------|
| Default   | 93.98%         | 97.56%              | 229.81秒 |
| Xavier    | **96.05%**     | **98.04%**          | 231.80秒 |
| Kaiming   | **96.35%**     | **98.00%**          | 195.80秒 |

## 核心发现

### 1. 第1轮准确率差异显著

- **Kaiming最高（96.35%）** → 说明Kaiming + ReLU的组合让信号从第一轮就传播稳定
- **Xavier次之（96.05%）** → Xavier也不错，但针对ReLU不如Kaiming优化
- **Default最低（93.98%）** → 默认初始化在这个场景下不是最优

**验证理论：** 好的初始化 = 信号方差稳定 = 梯度有效 = 第1轮就能快速学习

### 2. 最终准确率：Xavier略胜

- **Xavier: 98.04%**（最高）
- Kaiming: 98.00%（接近）
- Default: 97.56%（落后0.44-0.48个百分点）

**为什么Xavier最终略胜？**
- Kaiming只照顾前向传播（`2/n_in`）→ 起步快
- Xavier平衡前向+反向传播（`2/(n_in+n_out)`）→ 长期训练更稳定

**为什么差距不大？**
- 网络不够深（只有5层）
- ReLU的梯度特性好（正数区导数=1，反向传播不易消失）
- 训练轮数够多（10轮足够收敛）

### 3. 训练速度：Kaiming最快

- Kaiming: 195.80秒（最快）
- Default/Xavier: ~230秒

**原因：** Kaiming的初始化让信号传播最稳定 → 梯度下降路径更直接 → 收敛更快

## 理论验证

### Xavier初始化（适用Sigmoid/Tanh）

**公式：** `权重方差 = 2/(n_in + n_out)`

**设计思想：**
- 平衡前向传播（考虑n_in）和反向传播（考虑n_out）
- 让激活值和梯度的方差在多层传播时保持稳定
- 假设激活函数在0附近近似线性（Sigmoid/Tanh在0附近的特性）

**为什么不完美适配ReLU：**
- ReLU砍掉负数 → 方差减半
- Xavier没有补偿这个损失 → 深层网络信号会逐层衰减

### Kaiming初始化（适用ReLU）

**公式：** `权重方差 = 2/n_in`

**设计思想：**
- 优先保证前向传播稳定（只考虑n_in，不考虑n_out）
- 系数是2（而不是1）是因为：**补偿ReLU砍掉负数导致的方差减半**
- 反向传播靠ReLU的梯度特性（正数区导数=1）+ 其他技术（BatchNorm/残差连接）补位

**为什么适配ReLU：**
- ReLU砍掉负数 → 方差减半
- Kaiming提前翻倍方差（`2/n_in` 而不是 `1/n_in`）→ 抵消损失
- 前向传播信号稳定 → 第1轮就能快速学习

## 实战建议

**如何选择初始化方法？**

| 场景 | 激活函数 | 网络深度 | 推荐初始化 |
|------|---------|---------|----------|
| 现代深度网络 | ReLU / Leaky ReLU / PReLU | >10层 | **Kaiming** |
| 传统浅层网络 | Sigmoid / Tanh | <5层 | **Xavier** |
| 不确定 | - | - | **Kaiming**（更通用） |

**PyTorch实现：**
```python
import torch.nn as nn

# Xavier初始化
nn.init.xavier_uniform_(layer.weight)

# Kaiming初始化
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

# 偏置初始化
nn.init.zeros_(layer.bias)
```

## 核心洞察

1. **权重初始化的本质**：控制信号（激活值/梯度）的方差在多层传播时保持稳定
2. **为什么必须随机**：破缺对称性，让每个神经元学到不同特征
3. **为什么不能乱随机**：方差太大→爆炸，方差太小→消失
4. **Xavier vs Kaiming**：前者平衡前向+反向，后者优先前向+补偿ReLU
5. **实验验证**：Kaiming起步快（第1轮高），Xavier长期稳（最终略胜）

## 项目结构

```
weight_initialization/
├── model.py          # 5层全连接网络定义
├── init_methods.py   # 三种初始化方法实现
├── train.py          # 训练脚本
├── results.json      # 实验结果（训练历史）
└── README.md         # 本文件
```

## 如何运行

```bash
# 安装依赖
pip install torch torchvision

# 运行实验（训练三种初始化方法）
python train.py

# 查看结果
cat results.json
```

## 相关理论

- **Xavier初始化论文**：Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)
- **Kaiming初始化论文**：Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (He et al., 2015)
- **核心公式推导**：基于信号方差在前向/反向传播中的连锁反应
