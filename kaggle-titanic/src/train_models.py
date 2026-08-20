"""
Baseline 模型对比

目标：对比三个算法在 Titanic 数据集上的表现
- 逻辑回归（Logistic Regression）
- 随机森林（Random Forest）
- XGBoost

评估指标：准确率（Accuracy）
"""

import sys
sys.path.append('..')

from src.preprocessing import load_data, preprocess_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


def prepare_data():
    """
    准备训练数据

    返回：
    - X_train, X_val, y_train, y_val
    """
    # 加载并预处理数据
    train, test = load_data()
    train_processed, test_processed = preprocess_pipeline(train, test)

    # 分离特征和标签
    X = train_processed.drop('Survived', axis=1)
    y = train_processed['Survived']

    # TODO 1: 划分训练集和验证集
    # 提示：用 train_test_split，test_size=0.2，random_state=42（保证可复现）

    # ========== 你的代码开始 ==========
    # X_train, X_val, y_train, y_val = ...
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # ========== 你的代码结束 ==========

    return X_train, X_val, y_train, y_val


def train_logistic_regression(X_train, y_train):
    """
    训练逻辑回归模型

    超参数：
    - max_iter=1000（防止不收敛）
    - random_state=42
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    训练随机森林模型

    超参数：
    - n_estimators=100（树的数量）
    - random_state=42
    """
    # TODO 2: 创建并训练随机森林模型
    # 提示：RandomForestClassifier(n_estimators=100, random_state=42)
    #      然后 .fit(X_train, y_train)

    # ========== 你的代码开始 ==========
    # model = ...
    # model.fit(...)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # ========== 你的代码结束 ==========

    return model


def train_xgboost(X_train, y_train):
    """
    训练 XGBoost 模型

    超参数：
    - n_estimators=100
    - learning_rate=0.1
    - random_state=42
    """
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val, model_name):
    """
    评估模型性能

    输出：
    - 准确率
    - 混淆矩阵
    - 分类报告
    """
    # TODO 3: 预测 + 计算准确率
    # 提示：y_pred = model.predict(X_val)
    #      accuracy = accuracy_score(y_val, y_pred)

    # ========== 你的代码开始 ==========
    # y_pred = ...
    # accuracy = ...

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    # ========== 你的代码结束 ==========

    print(f"\n{'='*50}")
    print(f"{model_name} 评估结果")
    print(f"{'='*50}")
    print(f"准确率: {accuracy:.4f}")
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y_val, y_pred))
    print(f"\n分类报告:")
    print(classification_report(y_val, y_pred, target_names=['Died', 'Survived']))

    return accuracy


def main():
    """主流程：训练三个模型并对比"""
    print("加载数据...")
    X_train, X_val, y_train, y_val = prepare_data()
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 训练三个模型
    print("\n训练逻辑回归...")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_acc = evaluate_model(lr_model, X_val, y_val, "逻辑回归")

    print("\n训练随机森林...")
    rf_model = train_random_forest(X_train, y_train)
    rf_acc = evaluate_model(rf_model, X_val, y_val, "随机森林")

    print("\n训练 XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_acc = evaluate_model(xgb_model, X_val, y_val, "XGBoost")

    # 对比结果
    print(f"\n{'='*50}")
    print("模型对比总结")
    print(f"{'='*50}")
    results = pd.DataFrame({
        '模型': ['逻辑回归', '随机森林', 'XGBoost'],
        '准确率': [lr_acc, rf_acc, xgb_acc]
    })
    results = results.sort_values('准确率', ascending=False)
    print(results.to_string(index=False))
    print(f"\n最佳模型: {results.iloc[0]['模型']} (准确率: {results.iloc[0]['准确率']:.4f})")


if __name__ == '__main__':
    main()
