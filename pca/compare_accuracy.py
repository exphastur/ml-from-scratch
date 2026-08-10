# PCA 降维前后分类准确率对比实验
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ========== Part 1: 加载数据 ==========
iris = load_iris()
X = iris.data          # 原始 4 维特征
y = iris.target        # 标签

print("原始数据维度:", X.shape)
print()

# ========== Part 2: 划分训练集和测试集 ==========
# 将数据划分为训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)
print()

# ========== Part 3: 实验A - 用原始 4 维数据训练分类器 ==========
print("=" * 50)
print("实验A: 原始 4 维数据")
print("=" * 50)

# 创建逻辑回归分类器
clf_original = LogisticRegression(max_iter=200)

# 训练分类器
clf_original.fit(X_train, y_train)

# 在测试集上预测
y_pred_original = clf_original.predict(X_test)

# 计算准确率
acc_original = accuracy_score(y_test, y_pred_original)
print(f"原始 4 维数据的测试准确率: {acc_original:.2%}")
print()

# ========== Part 4: 实验B - 用 PCA 降到 2 维后训练分类器 ==========
print("=" * 50)
print("实验B: PCA 降到 2 维")
print("=" * 50)

# PCA 降维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)  # 在训练集上 fit，然后 transform
X_test_pca = pca.transform(X_test)        # 测试集只 transform，不 fit

print("PCA 降维后训练集维度:", X_train_pca.shape)
print("主成分1 保留方差:", f"{pca.explained_variance_ratio_[0]:.2%}")
print("主成分2 保留方差:", f"{pca.explained_variance_ratio_[1]:.2%}")
print("总共保留方差:", f"{pca.explained_variance_ratio_.sum():.2%}")
print()

# 创建新的逻辑回归分类器
clf_pca = LogisticRegression(max_iter = 200)

# 用降维后的数据训练
clf_pca.fit(X_train_pca ,y_train)

# 在降维后的测试集上预测
y_pred_pca =clf_pca.predict(X_test_pca)

# 计算准确率
acc_pca = accuracy_score(y_test, y_pred_pca)
print(f"PCA 2 维数据的测试准确率: {acc_pca:.2%}")
print()

# ========== Part 5: 对比结果 ==========
print("=" * 50)
print("结果对比")
print("=" * 50)
print(f"原始 4 维准确率: {acc_original:.2%}")
print(f"PCA 2 维准确率:  {acc_pca:.2%}")
print(f"准确率差异:      {abs(acc_original - acc_pca):.2%}")
print()

if acc_pca >= acc_original * 0.95:
    print("✅ 结论：降维后准确率几乎没有下降（<5%）！")
    print("   说明 PCA 成功保留了分类所需的关键信息。")
else:
    print("⚠️  结论：降维后准确率明显下降。")
    print("   可能需要保留更多主成分。")
