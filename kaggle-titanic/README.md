# Kaggle Titanic 生存预测

阶段2.5收官项目 - 完整的机器学习全流程实战

## 项目目标

- 走完真实数据的完整流程：探索 → 清洗 → 特征工程 → 模型选择 → 调参 → 提交
- 串联阶段2.5学过的所有算法（决策树、随机森林、GBDT、XGBoost等）
- 建立"先诊断、再调整"的调参思维，而非瞎试碰运气
- 产出：公开 notebook + 完整复盘报告

## 数据说明

- **训练集**: 891条样本，12个特征
- **测试集**: 418条样本
- **目标**: 预测乘客是否生存（二分类问题）

## 项目结构

```
kaggle-titanic/
├── data/                    # Kaggle 数据集
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/               # Jupyter notebooks
│   └── 01_eda.ipynb        # 数据探索
├── src/                     # Python 脚本
│   ├── preprocessing.py     # 数据清洗 + 特征工程
│   ├── train_model.py       # 模型训练
│   └── predict.py           # 生成提交文件
├── requirements.txt
└── README.md
```

## 学习日志

### 2026-08-17 | 项目启动 + 数据探索

**今天完成**：
- 项目环境搭建
- 数据探索 notebook 创建

**下一步**：
- 运行 EDA notebook，观察数据特征和缺失值
- 设计特征工程策略
