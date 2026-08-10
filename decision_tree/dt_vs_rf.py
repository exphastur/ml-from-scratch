from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 分割数据（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ——— 决策树 ———
dt = DecisionTreeClassifier(random_state=42)  # 创建对象
dt.fit(X_train, y_train)  # 训练

dt_trainScore = accuracy_score(y_train, dt.predict(X_train))
dt_testScore = accuracy_score(y_test, dt.predict(X_test))

# ——— 随机森林 ———
ft = RandomForestClassifier(n_estimators=100 ,random_state=42)
ft.fit(X_train, y_train)

ft_trainScore = accuracy_score(y_train, ft.predict(X_train))
ft_testScore =accuracy_score(y_test, ft.predict(X_test))

print(f'Decision Tree 训练: {dt_trainScore:.3f} 测试: {dt_testScore:.3f}')
print(f'Random Forest  训练: {ft_trainScore:.3f} 测试: {ft_testScore:.3f}')
# ——————————————————————————————————————————

