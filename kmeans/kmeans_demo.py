# kmeans_demo.py - K-means 聚类实战 + 肘部法则
# 路径：~/ml-from-scratch/kmeans/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ── 1. 生成数据（脚手架已写）─────────────────────────────────
# make_blobs 生成 3 个清晰的簇，每簇 50 个点，2 个特征（便于可视化）
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)
print(f"数据形状: {X.shape}")  # 应该是 (150, 2)

# ── 2. K-means 聚类（你来写，3 行）──────────────────────────
# 提示：
#   KMeans(n_clusters=3, n_init=10, random_state=42)
#   .fit_predict(X) → 训练 + 预测一步完成，返回每个点的簇标签（0/1/2）
#   .cluster_centers_ → 训练完成后，从 km 里取出 3 个中心点的坐标

km = KMeans(n_clusters=3, n_init=10, random_state=42)  # 创建 KMeans 对象
labels = km.fit_predict(X)  # 训练 + 预测（一步）
centers = km.cluster_centers_  # 取出中心点坐标

# ── 3. 可视化聚类结果（脚手架已写）──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], X[:, 1], s=40, alpha=0.7)
axes[0].set_title("原始数据（无标签）")

axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, alpha=0.7)
axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='中心')
axes[1].set_title("K-means 结果 (K=3)")
axes[1].legend()

plt.tight_layout()
plt.savefig("kmeans_result.png")
print("聚类结果 → kmeans_result.png")

# ── 4. 肘部法则（你来写，3 行）──────────────────────────────
# 提示：for k in range(1, 11)，每次：
#   ① KMeans(n_clusters=k, n_init=10, random_state=42)
#   ② .fit(X)
#   ③ 把 km.inertia_（sklearn 里 WCSS 的属性名）append 进 wcss

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters =k, n_init=10, random_state=42)
    km.fit_predict(X)
    wcss.append(km.inertia_)

# ── 5. 可视化肘部曲线（脚手架已写）──────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("K（簇数）")
plt.ylabel("WCSS（组内距离平方和）")
plt.title("肘部法则 — 找拐点")
plt.tight_layout()
plt.savefig("elbow.png")
print("肘部曲线 → elbow.png")
print("看图说说：拐点在 K= 几？为什么？")
