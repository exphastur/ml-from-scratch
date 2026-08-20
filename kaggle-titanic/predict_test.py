"""
Kaggle Titanic 最终预测脚本
用调优后的最佳参数在全部训练数据上训练，预测 test.csv
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

train_df = pd.read_csv('data/train.csv')

# 1. FamilySize 特征
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

# 2. Age 缺失值分组填充
train_df['Age'] = train_df['Age'].fillna(
    train_df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
)

# 3. FamilyCategory 三分类
train_df['FamilyCategory'] = np.where(
    train_df['FamilySize'] == 1, 'Alone',
    np.where(train_df['FamilySize'] <= 4, 'Small', 'Large')
)

# 4. CabinDeck 提取
train_df['CabinDeck'] = train_df['Cabin'].fillna('Unknown').str[0]

# 5. Embarked 缺失值填充
train_df['Embarked'] = train_df['Embarked'].fillna('S')

# 6. One-Hot 编码
train_encoded = pd.get_dummies(
    train_df,
    columns=['Sex', 'Embarked', 'FamilyCategory', 'CabinDeck'],
    drop_first=True
)

# 7. 选择特征列
feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize'] + \
               [col for col in train_encoded.columns if col.startswith(('Sex_', 'Embarked_', 'FamilyCategory_', 'CabinDeck_'))]

X_train = train_encoded[feature_cols]
y_train = train_encoded['Survived']

print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")

model = XGBClassifier(n_estimators=50, max_depth=3, scale_pos_weight=1.0, random_state=42)
model.fit(X_train, y_train)

test_df = pd.read_csv('data/test.csv')
test_passenger_ids = test_df['PassengerId'].copy()  # 保存 PassengerId 用于提交

# 对 test.csv 做同样的特征工程
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

test_df['Age'] = test_df['Age'].fillna(
    test_df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
)

test_df['FamilyCategory'] = np.where(
    test_df['FamilySize'] == 1, 'Alone',
    np.where(test_df['FamilySize'] <= 4, 'Small', 'Large')
)

test_df['CabinDeck'] = test_df['Cabin'].fillna('Unknown').str[0]

# test.csv 的 Fare 可能有缺失值，填充为中位数
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

test_encoded = pd.get_dummies(
    test_df,
    columns=['Sex', 'Embarked', 'FamilyCategory', 'CabinDeck'],
    drop_first=True
)

# 对齐特征列（test 可能缺少某些 One-Hot 列）
X_test = test_encoded.reindex(columns=feature_cols, fill_value=0)

print(f"测试数据形状: X_test={X_test.shape}")

predictions = model.predict(X_test)
pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions}).to_csv('submission.csv', index=False)

print("✅ submission.csv 已生成！")
print("📊 下一步：上传到 Kaggle 查看排行榜分数")
