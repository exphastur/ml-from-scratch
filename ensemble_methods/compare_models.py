"""
GBDT/XGBoost vs RandomForest 对比实验
阶段2.5 第二课 —— 集成学习方法对比

实验目标：
  - 理解 Bagging（随机森林）vs Boosting（GBDT/XGBoost）的性能差异
  - 观察准确率、过拟合程度、训练时间的三向对比
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time

# ─────────────────────────────────────────────
# 1. 加载数据
# ─────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 2. 三个模型
# ─────────────────────────────────────────────
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GBDT':         GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost':      XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='mlogloss'),
}

# ─────────────────────────────────────────────
# 3. 训练 & 评估
# ─────────────────────────────────────────────
results = {}

for name, model in models.items():
    # 训练模型并计时
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # 预测并计算准确率
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # 计算过拟合程度
    overfit = train_acc - test_acc

    results[name] = {
        'train_acc': train_acc,
        'test_acc':  test_acc,
        'overfit':   overfit,
        'time':      train_time,
    }

# ─────────────────────────────────────────────
# 4. 打印结果
# ─────────────────────────────────────────────
print(f"\n{'模型':<15} {'训练准确率':<12} {'测试准确率':<12} {'过拟合':<10} {'训练时间(s)'}")
print("─" * 65)
for name, res in results.items():
    print(f"{name:<15} {res['train_acc']:<12.2%} {res['test_acc']:<12.2%} "
          f"{res['overfit']:<10.2%} {res['time']:.4f}")
