"""
朴素贝叶斯 vs 逻辑回归 vs 决策树 对比实验
数据集：SMS Spam Collection（垃圾短信分类）
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 准备数据（使用内置的简化版数据集）
def load_data():
    """加载垃圾短信数据"""
    # 简化版数据（真实数据集需要下载）
    data = {
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'] * 100,
        'message': [
            'Free entry to win cash prize',
            'Call me when you get this',
            'Congratulations you won a prize',
            'Can we meet for lunch tomorrow',
            'Click here for free money',
            'Hey how are you doing today',
            'Claim your reward now limited time',
            'See you at the meeting later'
        ] * 100
    }
    df = pd.DataFrame(data)
    # 转换标签：spam=1, ham=0
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

# 2. 特征提取
df = load_data()
X = df['message']
y = df['label']

# 使用 CountVectorizer 把文本转成词频向量
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("实验1：完整数据集对比（训练集 {} 条）".format(X_train.shape[0]))
print("=" * 60)

# 4. 训练三个模型
nb = MultinomialNB()
logistic_regression = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)

model_nb = nb.fit(X_train, y_train)
model_logistic = logistic_regression.fit(X_train, y_train)
model_decision_tree = decision_tree.fit(X_train, y_train)

pred_nb = model_nb.predict(X_test)
pred_logic = model_logistic.predict(X_test)
pred_decison = model_decision_tree.predict(X_test)

nb_acc = accuracy_score(y_test, pred_nb)
lr_acc = accuracy_score(y_test, pred_logic)
dt_acc = accuracy_score(y_test, pred_decison)
print(f"朴素贝叶斯准确率: {nb_acc:.4f}")
print(f"逻辑回归准确率:   {lr_acc:.4f}")
print(f"决策树准确率:     {dt_acc:.4f}")

print("\n" + "=" * 60)
print("实验2：小样本对比（只用 50 条训练）")
print("=" * 60)

# 5. 小样本实验
X_train_small = X_train[:50]
y_train_small = y_train[:50]

nb_small = MultinomialNB()
logistic_regression_small = LogisticRegression(max_iter=1000)
decision_tree_small = DecisionTreeClassifier(random_state=42)

small_model_nb = nb_small.fit(X_train_small, y_train_small)
small_model_logic = logistic_regression_small.fit(X_train_small, y_train_small)
small_model_decision = decision_tree_small.fit(X_train_small, y_train_small)

pred_nb = nb_small.predict(X_test)
pred_logic = logistic_regression_small.predict(X_test)
pred_decison = decision_tree_small.predict(X_test)

nb_acc_small = accuracy_score(y_test, pred_nb)
lr_acc_small = accuracy_score(y_test, pred_logic)
dt_acc_small = accuracy_score(y_test, pred_decison)

print(f"朴素贝叶斯准确率: {nb_acc_small:.4f}")
print(f"逻辑回归准确率:   {lr_acc_small:.4f}")
print(f"决策树准确率:     {dt_acc_small:.4f}")

print("\n核心洞察：")
print("- 完整数据集：逻辑回归/决策树通常更强")
print("- 小样本：朴素贝叶斯更稳定（因为只需要统计词频）")
