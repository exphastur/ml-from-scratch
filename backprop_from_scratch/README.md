# 反向传播完全手写实现（numpy vs PyTorch对账验证）

## 项目目标

用 numpy 纯手写 2 层神经网络的完整反向传播，和 PyTorch 的 autograd 逐层对账，验证实现正确性。

**这是阶段 3「深度学习基础」补课 1-Part5 的实战验证，也是未来 GATE 硬关卡（从零手写反向传播+技术博客）的预演。**

---

## 网络结构

```
输入层 (2 特征)
    ↓  W1(2×3) + b1
隐藏层 (3 神经元, ReLU)
    ↓  W2(3×1) + b2
输出层 (1 神经元, Sigmoid)
    ↓
损失 (Binary Cross-Entropy)
```

**数据集：** 4 个样本，2 个特征，二分类任务

---

## 核心实现

### 前向传播
```python
Z1 = X @ W1 + b1
A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2
A2 = Sigmoid(Z2)
L = -mean[y·log(A2) + (1-y)·log(1-A2)]
```

### 反向传播（手写梯度）

**7 个梯度公式（从后往前）：**

```python
# 输出层
dZ2 = (A2 - y) / n                      # ∂L/∂Z2
dW2 = A1.T @ dZ2                        # ∂L/∂W2
db2 = np.sum(dZ2, axis=0, keepdims=True)  # ∂L/∂b2

# 梯度回传
dA1 = dZ2 @ W2.T                        # ∂L/∂A1

# 隐藏层
relu_grad = (Z1 > 0).astype(float)      # ReLU 导数
dZ1 = dA1 * relu_grad                   # ∂L/∂Z1
dW1 = X.T @ dZ1                         # ∂L/∂W1
db1 = np.sum(dZ1, axis=0, keepdims=True)  # ∂L/∂b1
```

---

## 对账结果

**numpy 手写梯度 vs PyTorch autograd 的最大差异：**

```
W2: 1.7e-9
b2: 1.4e-8
W1: 1.6e-9
b1: 1.1e-9
```

**全部 < 1e-6（阈值），对账成功！✅**

---

## 核心洞察

1. **链式法则的机械化执行**
   - 反向传播就是链式法则 + 矩阵乘法的组合
   - 从 Loss 开始，逐层往回算：∂L/∂Z2 → ∂L/∂W2 → ∂L/∂A1 → ∂L/∂Z1 → ∂L/∂W1

2. **ReLU 的导数超简单**
   - 正数区：导数 = 1
   - 负数区：导数 = 0
   - 代码：`(Z1 > 0).astype(float)`

3. **维度匹配是关键**
   - `A1.T @ dZ2`：(3,4) @ (4,1) = (3,1) ✅ 和 W2 形状一致
   - `X.T @ dZ1`：(2,4) @ (4,3) = (2,3) ✅ 和 W1 形状一致
   - 转置不是随便加的，是为了让矩阵乘法的维度对得上

4. **PyTorch 的 autograd 内部就在做这些事**
   - 你的 numpy 代码 = PyTorch autograd 的简化版
   - 工业级框架多了：动态计算图、内存优化、GPU 加速、自动微分引擎

---

## 运行方式

```bash
cd ~/ml-from-scratch/backprop_from_scratch
python backprop_numpy.py
```

**输出：** 前向传播结果 + numpy 梯度 + PyTorch 梯度 + 对账差异

---

## 学习收获

- ✅ 理解反向传播的完整推导（链式法则 + 矩阵求导）
- ✅ 手写实现并验证正确性（numpy vs PyTorch 对账）
- ✅ 理解 PyTorch autograd 的工作原理（动态计算图 + 自动求导）
- ✅ 掌握梯度的物理含义（参数对损失的敏感度）
- ✅ 为阶段 3 GATE 硬关卡做好准备

---

## 相关资源

- **补课 1-Part1~4**：反向传播理论（效率/计算图/链式法则/autograd机制）
- **阶段 3 GATE**：从零手写反向传播 + 公开技术博客
- **GitHub 仓库**：https://github.com/exphastur/ml-from-scratch

---

**日期：** 2026-09-28  
**状态：** ✅ 完成
