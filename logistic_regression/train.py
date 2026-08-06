"""
逻辑回归 - 从零手写实现
目标: 用纯 numpy 实现二分类逻辑回归，理解梯度下降完整训练循环
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# 1. 数据准备
# ============================================================

def load_data():
    """加载鸢尾花数据集，只取前两类做二分类"""
    iris = load_iris()
    # 只取前 100 个样本（前两类：setosa=0, versicolor=1）
    X = iris.data[:100]  # shape: (100, 4)
    y = iris.target[:100]  # shape: (100,)

    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=42
    # )
    x_temp, x_test, y_temp, y_test = train_test_split(
        X , y ,test_size = 0.2, random_state = 42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size = 0.25, random_state = 42)


    return x_train, x_test, y_train, y_test, x_val, y_val


# ============================================================
# 2. 模型组件（我写好的部分）
# ============================================================

def sigmoid(z):
    """
    Sigmoid 激活函数: σ(z) = 1 / (1 + e^(-z))
    输入: z 可以是标量、向量或矩阵
    输出: 同样形状，值域 [0, 1]
    """
    return 1 / (1 + np.exp(-z))


def forward(X, w, b):
    """
    前向传播: 计算预测概率

    输入:
        X: shape (n_samples, n_features) - 输入特征
        w: shape (n_features,) - 权重向量
        b: 标量 - 偏置

    输出:
        predictions: shape (n_samples,) - 每个样本的预测概率 [0,1]
    """
    z = np.dot(X, w) + b  # shape: (n_samples,)
    predictions = sigmoid(z)
    return predictions


# ============================================================
# 3. TODO: 你要实现的核心函数
# ============================================================

def compute_loss(y_true, y_pred):
    """
    TODO: 计算交叉熵损失

    公式: Loss = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]

    输入:
        y_true: shape (n_samples,) - 真实标签，0 或 1
        y_pred: shape (n_samples,) - 预测概率，[0, 1]

    输出:
        loss: 标量 - 平均交叉熵损失

    提示:
        - numpy 的 log 函数: np.log()
        - 为了数值稳定，可以给 log 里加一个极小值: np.log(y_pred + 1e-15)
          防止 log(0) 导致 -inf
        - 记得除以样本数 n
    """
    # ===== 你的代码开始 =====
    term = y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)
    _sumTerm = np.sum(term)
    _result = -_sumTerm /len(y_true)
    return _result

    # ===== 你的代码结束 =====


def compute_gradients(X, y_true, y_pred):
    """
    TODO: 计算梯度

    理论推导（交叉熵 + Sigmoid 的魔法）:
        ∂Loss/∂w = 1/n * X^T · (y_pred - y_true)
        ∂Loss/∂b = 1/n * Σ(y_pred - y_true)

    输入:
        X: shape (n_samples, n_features)
        y_true: shape (n_samples,)
        y_pred: shape (n_samples,)

    输出:
        grad_w: shape (n_features,) - w 的梯度
        grad_b: 标量 - b 的梯度

    提示:
        - 先算误差: error = y_pred - y_true
        - grad_w = (1/n) * X.T @ error  (矩阵乘法)
        - grad_b = (1/n) * np.sum(error)
    """
    n_samples = X.shape[0]

    # ===== 你的代码开始 =====
    n = len(y_true)
    grad_w = (X.T @ (y_pred - y_true))/n
    grad_b = np.sum(y_pred - y_true)/n
    return grad_w, grad_b

    # ===== 你的代码结束 =====


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

    # 初始化参数（随机小值）
    w = np.random.randn(n_features) * 0.01
    b = 0.0

    loss_history = []

    for epoch in range(epochs):
        # 1. 前向传播
        predictions = forward(X_train, w, b)

        # 2. 计算损失
        loss = compute_loss(y_train, predictions)
        loss_history.append(loss)

        # 3. 计算梯度
        grad_w, grad_b = compute_gradients(X_train, y_train, predictions)

        # 4. TODO: 更新参数（梯度下降）
        # 公式: w_new = w_old - learning_rate * grad_w
        #      b_new = b_old - learning_rate * grad_b
        # ===== 你的代码开始 =====
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        # ===== 你的代码结束 =====

        # 每 100 轮打印一次
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    return w, b, loss_history


# ============================================================
# 5. 评估与可视化（我写好的）
# ============================================================

def predict(X, w, b, threshold=0.5):
    """预测类别（0 或 1）"""
    probabilities = forward(X, w, b)
    return (probabilities >= threshold).astype(int)


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


# ===================================== =======================
# 6. 主函数
#   1. load_data       → 准备80训练+20测试
#   2. train           → 1000轮训练，得到w、b
#      ├─ forward      → 预测概率
#      ├─ compute_loss → 算损失
#      ├─ compute_gradients → 算梯度
#      └─ 更新参数     → w, b 每轮变化
#   3. predict         → 概率→类别
#   4. accuracy_score  → 算准确率
#   5. confusion_matrix → 看详细分类
#   6. plot_loss       → 画曲线
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("逻辑回归 - 从零手写实现")
    print("=" * 60)

    # 加载数据
    X_train, X_test, y_train, y_test, x_val ,y_val = load_data()
    print(f"\n训练集: {X_train.shape},验证集： {x_val.shape} 测试集: {X_test.shape}")
    # 训练模型
    print("\n开始训练...")
    w, b, loss_history = train(X_train, y_train, learning_rate=0.1, epochs=1000)

    # 使用验证集选择合适的学习率
    w_a, b_a, _ = train(X_train, y_train , learning_rate=0.1, epochs=1000)
    y_val_pred_a = predict(x_val, w_a, b_a)
    val_acc_a = accuracy_score(y_val ,y_val_pred_a)

    w_b, b_b, _ = train(X_train, y_train , learning_rate=0.01, epochs=1000)
    y_val_pred_b = predict(x_val, w_b, b_b)
    val_acc_b = accuracy_score(y_val ,y_val_pred_b)

    if(val_acc_a > val_acc_b):
        print('选择了0.1学习率的A方案')
        w_best = w_a
        b_best = b_a
    else:
        print('选择了0.01学习率的B方案')
        w_best = w_b
        b_best = b_b
    # 评估
    print("\n评估模型...")
  #  y_pred_train = predict(X_train, w, b)
  #  y_pred_test = predict(X_test, w, b)
 #   train_acc = accuracy_score(y_train, y_pred_train)
  #  test_acc = accuracy_score(y_test, y_pred_test)
    y_pred_test = predict(X_test ,w_best, b_best)
    test_acc = accuracy_score(y_test, y_pred_test)
   # print(f"训练集准确率: {train_acc:.2%}")
    print(f"测试集准确率: {test_acc:.2%}")

    print("\n混淆矩阵（测试集）:")
    print(confusion_matrix(y_test, y_pred_test))

    # 画损失曲线
    plot_loss(loss_history)

    #5折交叉验证
    print("\n===== 5折交叉验证（学习率0.01）=====")
    from sklearn.model_selection import KFold
    # 重新加载完整数据集  不再区分训练集 验证集
    iris = load_iris()
    x_full = iris.data[:100]
    y_full = iris.target[:100]
    fold_accuracies = []
    kfold = KFold(n_splits=5, shuffle= True ,random_state=42)
    for fold, (train_index, val_index) in enumerate(kfold.split(x_full), 1):
        X_train_fold, X_val = x_full[train_index], x_full[val_index]
        y_train_fold, y_val = y_full[train_index], y_full[val_index]
        w_fold,b_fold,_ = train(X_train_fold, y_train_fold, learning_rate=0.01, epochs=1000)
        y_val_pred = predict(x_val, w_fold, b_fold)
        val_acc = accuracy_score(y_val, y_val_pred)
        fold_accuracies.append(val_acc)
        print(f'第{fold}折 验证集准确率是：{val_acc:.2%}')
    mean_acc = np.mean(fold_accuracies)
    print(f"\n5折交叉验证平均准确率: {mean_acc:.2%}")
    print("\n训练完成！")
