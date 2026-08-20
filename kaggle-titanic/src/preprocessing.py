"""
数据预处理 + 特征工程

核心任务：
1. 处理缺失值
2. 创建新特征
3. 编码类别特征
"""

import pandas as pd
import numpy as np


def load_data():
    """加载训练集和测试集"""
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    return train, test


def handle_missing_values(df):
    """
    处理缺失值

    策略：
    - Age: 根据 Pclass + Sex 分组填充中位数
    - Embarked: 用众数填充（只缺2条）
    - Fare: 用中位数填充（测试集有1条缺失）
    - Cabin: 暂不处理（后面会提取首字母）
    """
    # TODO 1: 填充 Age 缺失值
    # 提示：用 df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    # 这样可以保留原有的 Age，只填充缺失的部分

    # ========== 你的代码开始 ==========
    df['Age'] = df['Age'].fillna(df.groupby(['Pclass','Sex'])['Age'].transform('median'))
    # ========== 你的代码结束 ==========

    # Embarked 填充众数
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fare 填充中位数（测试集有1条缺失）
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    return df


def create_features(df):
    """
    创建新特征

    新特征：
    1. FamilySize: 家庭规模（SibSp + Parch + 1）
    2. FamilyCategory: 家庭规模分类（Alone / Small / Large）
    3. CabinDeck: 船舱甲板（提取 Cabin 首字母，缺失标记为 'Unknown'）
    4. Title: 从姓名中提取称谓（Mr, Miss, Mrs 等）
    """
    # 1. 家庭规模
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # TODO 2: 创建家庭规模分类特征
    # 提示：根据你的观察，FamilySize=1 是 Alone，2-4 是 Small，5+ 是 Large
    # 可以用 np.where() 或者 pd.cut() 或者 apply() + lambda

    # ========== 你的代码开始 ==========

    df['FamilyCategory'] = np.where(df['FamilySize'] == 1, 'Alone', np.where(df['FamilySize'] <= 4 ,'Small', 'Large'))

    # ========== 你的代码结束 ==========

    # TODO 3: 提取 Cabin 首字母
    # 提示：Cabin='C85' 提取首字母 'C'，缺失值标记为 'Unknown'
    # 可以用 df['Cabin'].str[0]，但要先处理缺失值

    # ========== 你的代码开始 ==========

    df['CabinDeck'] = df['Cabin'].fillna('Unknown').str[0]

    # ========== 你的代码结束 ==========

    # 4. 提取称谓（这个我帮你写，涉及正则表达式）
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # 合并稀有称谓
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                         'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                         'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    return df


def encode_features(df):
    """
    编码类别特征

    策略：
    - Sex: male=1, female=0
    - Embarked: One-Hot 编码
    - FamilyCategory: One-Hot 编码
    - CabinDeck: One-Hot 编码
    - Title: One-Hot 编码
    """
    # Sex 编码
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # One-Hot 编码（会自动处理多个类别特征）
    df = pd.get_dummies(df, columns=['Embarked', 'FamilyCategory', 'CabinDeck', 'Title'],
                        drop_first=False)  # 不删除第一列，保留所有信息

    return df


def select_features(df):
    """
    选择最终用于建模的特征

    删除不需要的列：
    - PassengerId, Name, Ticket, Cabin（已提取信息）
    - SibSp, Parch（已合并为 FamilySize）
    """
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    # 只删除存在的列
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def preprocess_pipeline(train, test):
    """
    完整的预处理流程

    步骤：
    1. 处理缺失值
    2. 创建新特征
    3. 编码类别特征
    4. 选择特征
    """
    # 合并训练集和测试集（保证特征一致性）
    train['Dataset'] = 'train'
    test['Dataset'] = 'test'
    combined = pd.concat([train, test], axis=0, sort=False)

    # 执行预处理
    combined = handle_missing_values(combined)
    combined = create_features(combined)
    combined = encode_features(combined)
    combined = select_features(combined)

    # 分离训练集和测试集
    train_processed = combined[combined['Dataset'] == 'train'].drop('Dataset', axis=1)
    test_processed = combined[combined['Dataset'] == 'test'].drop('Dataset', axis=1)

    return train_processed, test_processed


if __name__ == '__main__':
    # 测试预处理流程
    train, test = load_data()
    print("原始数据形状:")
    print(f"训练集: {train.shape}")
    print(f"测试集: {test.shape}")

    train_processed, test_processed = preprocess_pipeline(train, test)

    print("\n处理后数据形状:")
    print(f"训练集: {train_processed.shape}")
    print(f"测试集: {test_processed.shape}")

    print("\n处理后特征列:")
    print(train_processed.columns.tolist())

    print("\n缺失值检查:")
    print(train_processed.isnull().sum()[train_processed.isnull().sum() > 0])
