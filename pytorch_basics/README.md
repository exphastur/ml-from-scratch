# PyTorch 工程基础（Checkpoint + 学习率调度 + 早停）

- **路径**：`pytorch_basics/`
- **时间**：2026-10-14
- **实现**：在 `weight_initialization` 的网络基础上，补齐 PyTorch 工程化的三件套
- **数据集**：MNIST，5层全连接（784→512→256→128→64→10）+ ReLU
- **优化器**：Adam(lr=1e-3) + ReduceLROnPlateau(patience=3, factor=0.5)

## 实现内容

### 1. Checkpoint 保存/加载（本次手写的核心）

```python
def save_checkpoint(model, optimizer, epoch, best_val_loss, path='checkpoint.pth'):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, path)
```

**为什么 optimizer 状态也要存？** Adam 内部维护 m_t（梯度EMA）和 v_t（梯度平方EMA）。
只存模型权重的话，续训时优化器从零起步，前几轮会剧烈震荡。

### 2. 学习率调度

`ReduceLROnPlateau`：验证 loss 连续 3 个 epoch 没改善 → lr × 0.5。

### 3. 早停（Early Stopping）

连续 5 个 epoch 验证 loss 没改善就停止训练。

## 实验结果

续训功能验证（三次连续运行）：

| 运行 | 起始轮次 | 结束轮次 | best_val_loss |
|------|---------|---------|---------------|
| 第 1 次 | 0（从头） | 2 | 0.0916 |
| 第 2 次 | **2（恢复）** | 5 | 0.0698 |
| 第 3 次 | **4（恢复）** | 5 | 0.0652 |

最终验证准确率 **98.1%**，证明 checkpoint 能正确恢复模型权重 + 优化器状态 + 训练进度。

## 核心收获

- **`zero_grad()` 的必要性**：`backward()` 默认是**累加**梯度到 `.grad`（而非覆盖），所以每个 batch 开始前必须清零，否则会把上一批的梯度混进来。这个设计是为了支持梯度累加技巧（多个小 batch 合并等价大 batch）。
- **Checkpoint 必须存 optimizer 状态**：Adam 内部维护 m_t（梯度 EMA）和 v_t（梯度平方 EMA），只存模型权重的话，续训时优化器从零起步会失去"惯性"，导致前几轮剧烈震荡。
- **`.grad` 是内存属性不是文件**：每个参数 `Tensor` 上挂了一个 `.grad` 属性存梯度值，住在内存里，`backward()` 更新它，`zero_grad()` 清零它——速度比写文件快几个数量级。
- **`model.train()` / `model.eval()` 切换**：训练模式下 Dropout 随机丢弃神经元、BatchNorm 用当前 batch 统计量；评估模式下 Dropout 关闭、BatchNorm 用训练时积累的全局统计量。
- **续训实验验证**：三次连续运行（从 ep0 → ep2 → ep5）证明 checkpoint 能正确恢复模型权重 + 优化器动量 + 训练进度，val_loss 从 0.0916 平滑改善到 0.0652。
- **全连接网络**：上一层每个神经元都连到下一层每个神经元（784 × 512 = 401,408 条连线），与卷积网络（局部连接）对比，是深度学习的基础网络结构。
- **张量（Tensor）**：n 维数组的统称（0维=标量、1维=向量、2维=矩阵、3维+=张量），PyTorch 用统一的 Tensor 类型承载所有维度，比 numpy 多了 GPU 加速和自动求导能力。

## 运行

```bash
cd pytorch_basics
python train.py
```

重复运行 `train.py` 会自动从 `checkpoint.pth` 续训。删除该文件则从头开始。
