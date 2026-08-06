# ML From Scratch

从零手写机器学习算法，深入理解底层原理。

## 项目列表

### 1. Logistic Regression（逻辑回归）
- **路径**：`logistic_regression/`
- **实现**：纯 numpy 手写，不调 sklearn 训练 API
- **核心**：梯度下降 + 交叉熵损失 + Sigmoid 激活
- **数据集**：鸢尾花（Iris）二分类

## 环境

- Python 3.11
- numpy, matplotlib, scikit-learn（仅用于加载数据集和评估指标）

## 安装

```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## 运行

```bash
cd logistic_regression
python train.py
```
