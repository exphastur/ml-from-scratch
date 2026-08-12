"""
SVM 核函数魔法实验
对比线性 SVM vs 核函数 SVM 在非线性数据上的表现
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("=" * 60)
print("实验1：线性可分数据（Iris 前两类）")
print("=" * 60)

# 1. 加载 Iris 数据集，只取前两类（线性可分）
iris = datasets.load_iris()
X = iris.data[iris.target != 2][:, :2]  # 只用前两个特征（方便可视化）
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练线性 SVM
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
pred_svm_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, pred_svm_linear)
print(f"线性 SVM 准确率: {acc_linear:.4f}")

# 创建并训练核函数 SVM（RBF）
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
pred_svm_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, pred_svm_rbf)
print(f"核函数 SVM 准确率: {acc_rbf:.4f}")
print("结论：线性数据上，两者都能分开\n")

print("=" * 60)
print("实验2：非线性可分数据（同心圆）")
print("=" * 60)

# 2. 生成同心圆数据（经典的非线性可分场景）
X_circle, y_circle = datasets.make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_circle, y_circle, test_size=0.3, random_state=42
)

# 线性 SVM（同心圆数据）
SVC_circle_linear = SVC(kernel='linear')
SVC_circle_linear.fit(X_train_c, y_train_c)
acc_linear_c = accuracy_score(y_test_c, SVC_circle_linear.predict(X_test_c))
print(f"线性 SVM 准确率: {acc_linear_c:.4f}  <-- 分不开！")

# 核函数 SVM（同心圆数据）
SVC_circle_rbf = SVC(kernel='rbf')
SVC_circle_rbf.fit(X_train_c, y_train_c)
acc_rbf_c = accuracy_score(y_test_c, SVC_circle_rbf.predict(X_test_c))
print(f"核函数 SVM 准确率: {acc_rbf_c:.4f}  <-- 完美分开！")

print("\n" + "=" * 60)
print("核心洞察")
print("=" * 60)
print("1. 线性 SVM = 用直线/超平面分类，只能处理线性可分数据")
print("2. 核函数 SVM = 把数据映射到高维，用曲线分类")
print("3. RBF 核函数 = 最常用的核，能处理复杂的非线性边界")
print("4. 同心圆数据：线性 SVM 约 50%（随机猜），RBF SVM 接近 100%")
