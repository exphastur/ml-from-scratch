"""
超参数调优 - XGBoost

目标：通过调整超参数，让 XGBoost 超过逻辑回归的 83.24%

调优策略：
1. 先调 n_estimators（树的数量）
2. 再调 max_depth（树的深度，防止过拟合）
3. 最后调 scale_pos_weight（处理类别不平衡）
"""

import sys
sys.path.append('..')

from src.preprocessing import load_data, preprocess_pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


def prepare_data():
    """准备训练数据"""
    train, test = load_data()
    train_processed, test_processed = preprocess_pipeline(train, test)

    X = train_processed.drop('Survived', axis=1)
    y = train_processed['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


def check_class_balance(y_train):
    """
    检查类别平衡情况

    返回：
    - pos_count: Survived 的数量
    - neg_count: Died 的数量
    - ratio: neg/pos 的比例（用于 scale_pos_weight）
    """
    counts = y_train.value_counts()
    neg_count = counts[0]  # Died
    pos_count = counts[1]  # Survived
    ratio = neg_count / pos_count

    print(f"训练集类别分布:")
    print(f"  Died (0):     {neg_count} ({neg_count/len(y_train)*100:.1f}%)")
    print(f"  Survived (1): {pos_count} ({pos_count/len(y_train)*100:.1f}%)")
    print(f"  比例 (Died/Survived): {ratio:.2f}")
    print(f"\n建议的 scale_pos_weight: {ratio:.2f}\n")

    return pos_count, neg_count, ratio


def tune_n_estimators(X_train, y_train, X_val, y_val):
    """
    调优 n_estimators（树的数量）

    测试范围：50, 100, 150, 200, 300
    """
    print("="*50)
    print("调优 n_estimators（树的数量）")
    print("="*50)

    # TODO 1: 尝试不同的 n_estimators，找出最佳值
    # 提示：用 for 循环遍历 [50, 100, 150, 200, 300]
    #      每次创建 XGBClassifier(n_estimators=n, random_state=42, eval_metric='logloss')
    #      训练、预测、计算准确率，记录结果

    # ========== 你的代码开始 ==========
    results = []
    for n in [50, 100, 150, 200, 300]:
        # model = XGBClassifier(...)
        # model.fit(...)
        # y_pred = model.predict(...)
        # acc = accuracy_score(...)
        # results.append({'n_estimators': n, '准确率': acc})
        model = XGBClassifier(n_estimators=n ,random_state=42)
        model.fit(X_train ,y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        results.append({'n_estimators': n ,'准确率': acc})
    # ========== 你的代码结束 ==========

    # 输出结果
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    best = df.loc[df['准确率'].idxmax()]
    print(f"\n最佳 n_estimators: {int(best['n_estimators'])} (准确率: {best['准确率']:.4f})")
    return int(best['n_estimators'])


def tune_max_depth(X_train, y_train, X_val, y_val, best_n_estimators):
    """
    调优 max_depth（树的深度）

    测试范围：3, 4, 5, 6, 7
    固定 n_estimators 为上一步找到的最佳值
    """
    print("\n" + "="*50)
    print("调优 max_depth（树的深度）")
    print("="*50)

    # TODO 2: 尝试不同的 max_depth，找出最佳值
    # 提示：遍历 [3, 4, 5, 6, 7]
    #      XGBClassifier(n_estimators=best_n_estimators, max_depth=d, random_state=42, eval_metric='logloss')

    # ========== 你的代码开始 ==========
    results = []
    for d in [3, 4, 5, 6, 7]:
        model = XGBClassifier(n_estimators=best_n_estimators, max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        results.append({'max_depth': d, '准确率': acc})
    # ========== 你的代码结束 ==========

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    best = df.loc[df['准确率'].idxmax()]
    print(f"\n最佳 max_depth: {int(best['max_depth'])} (准确率: {best['准确率']:.4f})")
    return int(best['max_depth'])


def tune_scale_pos_weight(X_train, y_train, X_val, y_val, best_n_estimators, best_max_depth, ratio):
    """
    调优 scale_pos_weight（处理类别不平衡）

    测试范围：1.0（不调整）, ratio（建议值）, ratio*1.2, ratio*0.8
    """
    print("\n" + "="*50)
    print("调优 scale_pos_weight（类别不平衡权重）")
    print("="*50)

    # TODO 3: 尝试不同的 scale_pos_weight
    # 提示：遍历 [1.0, ratio, ratio*1.2, ratio*0.8]
    #      XGBClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth,
    #                    scale_pos_weight=w, random_state=42, eval_metric='logloss')

    # ========== 你的代码开始 ==========
    results = []
    for w in [1.0, ratio, ratio*1.2, ratio*0.8]:
        model = XGBClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, scale_pos_weight=w, random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        results.append({'scale_pos_weight': w, '准确率': acc})
    # ========== 你的代码结束 ==========

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    best = df.loc[df['准确率'].idxmax()]
    print(f"\n最佳 scale_pos_weight: {best['scale_pos_weight']:.2f} (准确率: {best['准确率']:.4f})")
    return best['scale_pos_weight']


def final_evaluation(X_train, y_train, X_val, y_val, best_params):
    """
    用最佳参数训练最终模型，输出详细评估
    """
    print("\n" + "="*50)
    print("最终模型评估")
    print("="*50)
    print(f"最佳超参数: {best_params}\n")

    model = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        scale_pos_weight=best_params['scale_pos_weight'],
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"准确率: {accuracy:.4f}")
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y_val, y_pred))

    # 对比 Baseline
    baseline_acc = 0.8324  # 逻辑回归的准确率
    if accuracy > baseline_acc:
        print(f"\n🎉 成功超越逻辑回归！提升了 {(accuracy - baseline_acc)*100:.2f} 个百分点")
    else:
        print(f"\n未能超越逻辑回归（{baseline_acc:.4f}），差距 {(baseline_acc - accuracy)*100:.2f} 个百分点")

    return model


def main():
    """主流程：逐步调优 XGBoost"""
    print("加载数据...")
    X_train, X_val, y_train, y_val = prepare_data()
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}\n")

    # 检查类别平衡
    pos_count, neg_count, ratio = check_class_balance(y_train)

    # 逐步调优
    best_n_estimators = tune_n_estimators(X_train, y_train, X_val, y_val)
    best_max_depth = tune_max_depth(X_train, y_train, X_val, y_val, best_n_estimators)
    best_scale_pos_weight = tune_scale_pos_weight(X_train, y_train, X_val, y_val,
                                                    best_n_estimators, best_max_depth, ratio)

    # 最终评估
    best_params = {
        'n_estimators': best_n_estimators,
        'max_depth': best_max_depth,
        'scale_pos_weight': best_scale_pos_weight
    }
    final_model = final_evaluation(X_train, y_train, X_val, y_val, best_params)


if __name__ == '__main__':
    main()
