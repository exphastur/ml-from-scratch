# PCA 降维实验 - Iris 数据集 4维 → 2维
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# ========== Part 1: 加载数据 ==========
# 加载 Iris 数据集
iris = load_iris()
X = iris.data          # 特征矩阵 (150, 4)：150个样本，4个特征
y = iris.target        # 标签 (150,)：0/1/2 三种花
feature_names = iris.feature_names
target_names = iris.target_names

print("原始数据维度:", X.shape)
print("特征名称:", feature_names)
print("类别名称:", target_names)
print()

# ========== Part 2: PCA 降维 ==========
# 创建 PCA 对象，降到 2 维
pca = PCA(n_components=2)

# 对数据 X 进行降维
X_pca = pca.fit_transform(X)

print("降维后数据维度:", X_pca.shape)
print()

# ========== Part 3: 查看方差占比（信息保留率）==========
# explained_variance_ratio_ 是 PCA 自动计算的：每个主成分保留了多少比例的方差
variance_ratio = pca.explained_variance_ratio_
print("主成分1 保留的方差占比:", f"{variance_ratio[0]:.2%}")
print("主成分2 保留的方差占比:", f"{variance_ratio[1]:.2%}")
print("总共保留的方差占比:", f"{variance_ratio.sum():.2%}")
print()

# ========== Part 3.5: 查看主成分的权重（含义）==========
# components_ 是 PCA 计算出的主成分权重矩阵 (2, 4)
# 每一行是一个主成分，每一列对应一个原始特征的权重
print("主成分的权重（系数）：")
print("特征:", feature_names)
print()
print("主成分1 的权重:", pca.components_[0])
print("主成分2 的权重:", pca.components_[1])
print()

# 解读主成分含义（看哪些特征的权重大）
print("主成分含义解读：")
pc1 = pca.components_[0]
pc2 = pca.components_[1]
print(f"主成分1 = {pc1[0]:.3f}×萼长 + {pc1[1]:.3f}×萼宽 + {pc1[2]:.3f}×瓣长 + {pc1[3]:.3f}×瓣宽")
print(f"主成分2 = {pc2[0]:.3f}×萼长 + {pc2[1]:.3f}×萼宽 + {pc2[2]:.3f}×瓣长 + {pc2[3]:.3f}×瓣宽")
print()

# ========== Part 4: 可视化降维后的数据 ==========
# 画散点图：横轴=主成分1，纵轴=主成分2，颜色=花的类别
plt.figure(figsize=(8, 6))

# 为三种花分别画点（用不同颜色）
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(target_names):
    # 找到属于第 i 类的样本
    indices = y == i
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                c=colors[i], label=target_name, alpha=0.6, edgecolors='k')

plt.xlabel(f'主成分1 ({variance_ratio[0]:.1%} 方差)')
plt.ylabel(f'主成分2 ({variance_ratio[1]:.1%} 方差)')
plt.title('PCA 降维可视化 - Iris 数据集 (4维 → 2维)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=150)
print("可视化图已保存为 pca_visualization.png")
plt.show()
