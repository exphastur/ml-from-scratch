"""
反向传播完全手写实现（numpy版）
目标：用numpy手写2层神经网络的前向+反向传播，和PyTorch对账验证
"""

import numpy as np

# ==================== 1. 数据准备 ====================
np.random.seed(42)

# 超小数据集（4个样本，2个特征）
X = np.array([
    [0.5, 0.8],
    [0.2, 0.3],
    [0.9, 0.7],
    [0.1, 0.4]
])  # shape: (4, 2)

y = np.array([[1], [0], [1], [0]])  # shape: (4, 1)

n_samples = X.shape[0]  # 4

# ==================== 2. 初始化参数 ====================
# 网络结构：输入2 → 隐藏层3 → 输出1
np.random.seed(42)
W1 = np.random.randn(2, 3) * 0.1  # (2, 3)
b1 = np.zeros((1, 3))              # (1, 3)
W2 = np.random.randn(3, 1) * 0.1  # (3, 1)
b2 = np.zeros((1, 1))              # (1, 1)

print("="*50)
print("初始化参数：")
print(f"W1 shape: {W1.shape}, W2 shape: {W2.shape}")
print(f"b1 shape: {b1.shape}, b2 shape: {b2.shape}")
print()


# ==================== 3. 前向传播 ====================
def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)

def sigmoid(z):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-z))

# 前向传播
Z1 = X @ W1 + b1           # (4, 3)
A1 = relu(Z1)              # (4, 3)
Z2 = A1 @ W2 + b2          # (4, 1)
A2 = sigmoid(Z2)           # (4, 1)

# 计算损失（二分类交叉熵）
epsilon = 1e-15  # 防止log(0)
loss = -np.mean(y * np.log(A2 + epsilon) + (1 - y) * np.log(1 - A2 + epsilon))

print("="*50)
print("前向传播：")
print(f"Z1 shape: {Z1.shape}, A1 shape: {A1.shape}")
print(f"Z2 shape: {Z2.shape}, A2 shape: {A2.shape}")
print(f"Loss: {loss:.6f}")
print()


# ==================== 4. 反向传播（你来填！）====================
"""
TODO: 用链式法则计算5个梯度

提示：
1. 先算 dZ2 = ∂L/∂Z2（从损失开始，往回推第一步）
2. 再算 dW2 = ∂L/∂W2 和 db2 = ∂L/∂b2
3. 然后算 dA1 = ∂L/∂A1（从输出层传到隐藏层的桥梁）
4. 再算 dZ1 = ∂L/∂Z1（注意：ReLU的导数！）
5. 最后算 dW1 = ∂L/∂W1 和 db1 = ∂L/∂b1

维度提示：
- dW2 应该和 W2 形状一样：(3, 1)
- dW1 应该和 W1 形状一样：(2, 3)
- 用 .T 转置，用 @ 做矩阵乘法
- 别忘了除以 n_samples（梯度要平均）
"""

# 反向传播：从后往前计算梯度
# 第一步：输出层的梯度
dZ2 = (A2 - y) / n_samples  # ∂L/∂Z2 = (预测 - 真实) / 样本数
dW2 = A1.T @ dZ2             # ∂L/∂W2 = A1^T @ dZ2（链式法则）
db2 = np.sum(dZ2, axis=0, keepdims=True)  # ∂L/∂b2 = dZ2沿样本求和

# 第二步：梯度传回隐藏层
dA1 = dZ2 @ W2.T             # ∂L/∂A1 = dZ2 @ W2^T（反向传播）

# 第三步：隐藏层的梯度
relu_grad = (Z1 > 0).astype(float)  # ReLU导数：Z1>0的地方为1，否则为0
dZ1 = dA1 * relu_grad        # ∂L/∂Z1 = dA1 * ReLU'(Z1)（逐元素乘法）
dW1 = X.T @ dZ1              # ∂L/∂W1 = X^T @ dZ1
db1 = np.sum(dZ1, axis=0, keepdims=True)  # ∂L/∂b1 = dZ1沿样本求和


print("="*50)
print("反向传播（numpy手写）：")
print(f"dW2:\n{dW2}")
print(f"db2: {db2}")
print(f"dW1:\n{dW1}")
print(f"db1: {db1}")
print()


# ==================== 5. PyTorch对照（自动求导）====================
import torch

# 转换为PyTorch张量
X_torch = torch.tensor(X, dtype=torch.float32, requires_grad=False)
y_torch = torch.tensor(y, dtype=torch.float32, requires_grad=False)

# 参数（需要梯度）
W1_torch = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
b1_torch = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
W2_torch = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
b2_torch = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

# 前向传播
Z1_torch = X_torch @ W1_torch + b1_torch
A1_torch = torch.relu(Z1_torch)
Z2_torch = A1_torch @ W2_torch + b2_torch
A2_torch = torch.sigmoid(Z2_torch)

# 损失
loss_torch = -torch.mean(y_torch * torch.log(A2_torch + 1e-15) +
                         (1 - y_torch) * torch.log(1 - A2_torch + 1e-15))

# 反向传播（自动）
loss_torch.backward()

print("="*50)
print("反向传播（PyTorch自动求导）：")
print(f"W2.grad:\n{W2_torch.grad.numpy()}")
print(f"b2.grad: {b2_torch.grad.numpy()}")
print(f"W1.grad:\n{W1_torch.grad.numpy()}")
print(f"b1.grad: {b1_torch.grad.numpy()}")
print()


# ==================== 6. 对账验证 ====================
"""
TODO: 解除注释，对比numpy梯度 vs PyTorch梯度
如果差异 < 1e-6，说明你的手写实现正确！
"""

print("="*50)
print("对账验证（numpy vs PyTorch）：")
diff_W2 = np.abs(dW2 - W2_torch.grad.numpy()).max()
diff_b2 = np.abs(db2 - b2_torch.grad.numpy()).max()
diff_W1 = np.abs(dW1 - W1_torch.grad.numpy()).max()
diff_b1 = np.abs(db1 - b1_torch.grad.numpy()).max()

print(f"W2 最大差异: {diff_W2:.10f}")
print(f"b2 最大差异: {diff_b2:.10f}")
print(f"W1 最大差异: {diff_W1:.10f}")
print(f"b1 最大差异: {diff_b1:.10f}")
print()

if max(diff_W2, diff_b2, diff_W1, diff_b1) < 1e-6:
    print("[SUCCESS] 对账成功！你的手写反向传播完全正确！")
else:
    print("[FAILED] 对账失败！检查梯度计算公式。")
