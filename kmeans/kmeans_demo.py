# kmeans_demo.py - K-means 聚类实战 + 肘部法则

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成数据：3 个清晰的簇，每簇 50 个点，2 个特征（便于可视化）
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)
print(f"数据形状: {X.shape}")

# K-means 聚类
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X)
centers = km.cluster_centers_

# 可视化聚类结果
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

# 肘部法则：遍历 K=1~10，记录每个 K 的 WCSS
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit_predict(X)
    wcss.append(km.inertia_)

# 可视化肘部曲线
plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("K（簇数）")
plt.ylabel("WCSS（组内距离平方和）")
plt.title("肘部法则 — 找拐点")
plt.tight_layout()
plt.savefig("elbow.png")
print("肘部曲线 → elbow.png")
