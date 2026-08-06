"""
逻辑回归 - 从零手写实现
目标: 用纯 numpy 实现二分类逻辑回归，理解梯度下降完整训练循环
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# 1. 数据准备
# ============================================================

def load_data():
    """加载鸢尾花数据集，只取前两类做二分类"""
    iris = load_iris()
    X = iris.data[:100]
    y = iris.target[:100]

    x_temp, x_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.25, random_state=42
    )
    return x_train, x_test, y_train, y_test, x_val, y_val


# ============================================================
# 2. 模型组件
# ============================================================

def sigmoid(z):
    """Sigmoid 激活函数: σ(z) = 1 / (1 + e^(-z))
    输入: z 可以是标量、向量或矩阵
    输出: 同样形状，值域 [0, 1]
    """
    return 1 / (1 + np.exp(-z))


def forward(X, w, b):
    """
    前向传播：计算预测概率

    输入:
        X: shape (n_samples, n_features) - 输入特征
        w: shape (n_features,) - 权重向量
        b: 标量 - 偏置
    输出:
        predictions: shape (n_samples,) - 每个样本的预测概率 [0, 1]
    """
    z = np.dot(X, w) + b
    return sigmoid(z)


# ============================================================
# 3. 核心函数
# ============================================================

def compute_loss(y_true, y_pred):
    """
    计算交叉熵损失
    公式: Loss = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]

    输入:
        y_true: shape (n_samples,) - 真实标签，0 或 1
        y_pred: shape (n_samples,) - 预测概率，[0, 1]
    输出:
        loss: 标量 - 平均交叉熵损失
    """
    term = y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)
    return -np.sum(term) / len(y_true)


def compute_gradients(X, y_true, y_pred):
    """
    计算梯度
    公式: ∂Loss/∂w = 1/n * X^T · (y_pred - y_true)
          ∂Loss/∂b = 1/n * Σ(y_pred - y_true)

    输入:
        X: shape (n_samples, n_features)
        y_true: shape (n_samples,)
        y_pred: shape (n_samples,)
    输出:
        grad_w: shape (n_features,) - w 的梯度
        grad_b: 标量 - b 的梯度
    """
    n = len(y_true)
    grad_w = (X.T @ (y_pred - y_true)) / n
    grad_b = np.sum(y_pred - y_true) / n
    return grad_w, grad_b


# ============================================================
# 4. 训练循环
# ============================================================

def train(X_train, y_train, learning_rate=0.1, epochs=1000):
    """
    训练逻辑回归模型

    输入:
        X_train: shape (n_samples, n_features)
        y_train: shape (n_samples,)
        learning_rate: 学习率
        epochs: 训练轮数
    输出:
        w, b: 训练好的参数
        loss_history: 每轮的损失值，用于画图
    """
    n_samples, n_features = X_train.shape
    w = np.random.randn(n_features) * 0.01
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        predictions = forward(X_train, w, b)
        loss = compute_loss(y_train, predictions)
        loss_history.append(loss)
        grad_w, grad_b = compute_gradients(X_train, y_train, predictions)

        # 梯度下降更新参数
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    return w, b, loss_history


# ============================================================
# 5. 评估与可视化
# ============================================================

def predict(X, w, b, threshold=0.5):
    """预测类别（0 或 1）"""
    return (forward(X, w, b) >= threshold).astype(int)


def plot_loss(loss_history):
    """画损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("损失曲线已保存到 loss_curve.png")


# ============================================================
# 6. 主函数
#   1. load_data           → 准备训练/验证/测试集（60/20/20）
#   2. train               → 1000轮训练，用验证集选最优学习率
#      ├─ forward          → 预测概率
#      ├─ compute_loss     → 算交叉熵损失
#      ├─ compute_gradients→ 算梯度
#      └─ 梯度下降         → 更新 w, b
#   3. predict             → 概率 → 类别
#   4. accuracy_score      → 测试集准确率
#   5. confusion_matrix    → 详细分类统计
#   6. KFold               → 5折交叉验证
#   7. plot_loss           → 画损失曲线
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("逻辑回归 - 从零手写实现")
    print("=" * 60)

    X_train, X_test, y_train, y_test, x_val, y_val = load_data()
    print(f"\n训练集: {X_train.shape}, 验证集: {x_val.shape}, 测试集: {X_test.shape}")

    # 用验证集选择最优学习率
    print("\n开始训练（选学习率）...")
    w_a, b_a, loss_history_a = train(X_train, y_train, learning_rate=0.1, epochs=1000)
    val_acc_a = accuracy_score(y_val, predict(x_val, w_a, b_a))

    w_b, b_b, loss_history_b = train(X_train, y_train, learning_rate=0.01, epochs=1000)
    val_acc_b = accuracy_score(y_val, predict(x_val, w_b, b_b))

    if val_acc_a > val_acc_b:
        print('选择了 0.1 学习率的方案')
        w_best, b_best, loss_history = w_a, b_a, loss_history_a
    else:
        print('选择了 0.01 学习率的方案')
        w_best, b_best, loss_history = w_b, b_b, loss_history_b

    # 测试集评估
    print("\n评估模型...")
    y_pred_test = predict(X_test, w_best, b_best)
    print(f"测试集准确率: {accuracy_score(y_test, y_pred_test):.2%}")
    print("\n混淆矩阵（测试集）:")
    print(confusion_matrix(y_test, y_pred_test))

    # 画损失曲线
    plot_loss(loss_history)

    # 5折交叉验证
    print("\n===== 5折交叉验证（学习率0.01）=====")
    iris = load_iris()
    x_full = iris.data[:100]
    y_full = iris.target[:100]
    fold_accuracies = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_full), 1):
        X_train_fold, X_val_fold = x_full[train_idx], x_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
        w_fold, b_fold, _ = train(X_train_fold, y_train_fold, learning_rate=0.01, epochs=1000)
        val_acc = accuracy_score(y_val_fold, predict(X_val_fold, w_fold, b_fold))  # 修复：用当前折的验证集
        fold_accuracies.append(val_acc)
        print(f'第{fold}折验证集准确率: {val_acc:.2%}')

    print(f"\n5折交叉验证平均准确率: {np.mean(fold_accuracies):.2%}")
    print("\n训练完成！")
