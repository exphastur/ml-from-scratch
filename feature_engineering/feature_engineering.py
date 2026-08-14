import numpy as np

# ==================== 模拟招聘数据 ====================
# 5个候选人的简历数据
candidates = [
    {"name": "张三", "education": "本科", "age": 28, "experience": 5, "salary": 15000, "expected": 25000},
    {"name": "李四", "education": "硕士", "age": 26, "experience": 3, "salary": 18000, "expected": 28000},
    {"name": "王五", "education": "博士", "age": 30, "experience": 2, "salary": 20000, "expected": 35000},
    {"name": "赵六", "education": "本科", "age": 35, "experience": 10, "salary": 22000, "expected": 30000},
    {"name": "离群值", "education": "硕士", "age": 50, "experience": 20, "salary": 50000, "expected": 1000000},  # 期望工资离群
]

print("==================== 原始数据 ====================")
for c in candidates:
    print(f"{c['name']}: 学历={c['education']}, 年龄={c['age']}, 经验={c['experience']}年, 期望={c['expected']}")

# ==================== 第1步：处理类别变量 - One-Hot编码 ====================
print("\n==================== One-Hot 编码（学历） ====================")

# 提取所有学历类别
educations = [c["education"] for c in candidates]
unique_educations = sorted(set(educations))  # ['博士', '本科', '硕士']
print(f"学历类别: {unique_educations}")

# One-Hot 编码实现
education_encoded = {}
for c in candidates:
    edu = c["education"]
    encoding = [0, 0, 0]
    number = unique_educations.index(edu)
    encoding[number] = 1
    education_encoded[c["name"]] = encoding

# ==================== 第2步：处理数值变量 - 离群值截断 ====================
print("\n==================== 离群值处理（期望工资） ====================")

# 提取期望工资
expected_salaries = [c["expected"] for c in candidates]
print(f"原始期望工资: {expected_salaries}")

# 计算均值和标准差
mean_salary = np.mean(expected_salaries)
std_salary = np.std(expected_salaries)
print(f"均值: {mean_salary:.0f}, 标准差: {std_salary:.0f}")

# 离群值截断实现
upper_limit = mean_salary + 3 * std_salary
lower_limit = max(0, mean_salary - 3 * std_salary)
print(f"合理范围: [{lower_limit:.0f}, {upper_limit:.0f}]")

clipped_salaries = []
for salary in expected_salaries:
    if salary > upper_limit:
        clipped_salaries.append(upper_limit)
    elif salary < lower_limit:
        clipped_salaries.append(lower_limit)
    else:
        clipped_salaries.append(salary)

print(f"截断后期望工资: {clipped_salaries}")

# ==================== 第3步：归一化 - Min-Max ====================
print("\n==================== Min-Max 归一化 ====================")

# 需要归一化的数值特征：年龄、工作经验、截断后的期望工资
ages = [c["age"] for c in candidates]
experiences = [c["experience"] for c in candidates]

print(f"年龄范围: {min(ages)} - {max(ages)}")
print(f"经验范围: {min(experiences)} - {max(experiences)}")
print(f"期望工资范围: {min(clipped_salaries)} - {max(clipped_salaries)}")

# Min-Max 归一化实现
normalized_ages = []
normalized_experiences = []
normalized_salaries = []

for age in ages:
    normalized_ages.append((age - min(ages)) / (max(ages) - min(ages)))

for experience in experiences:
    normalized_experiences.append((experience - min(experiences)) / (max(experiences) - min(experiences)))

for salary in clipped_salaries:
    normalized_salaries.append((salary - min(clipped_salaries)) / (max(clipped_salaries) - min(clipped_salaries)))

# 打印归一化结果
print("\n归一化后的特征（前3个候选人）：")
for i in range(min(3, len(candidates))):
    print(f"{candidates[i]['name']}: 年龄={normalized_ages[i]:.3f}, 经验={normalized_experiences[i]:.3f}, 期望工资={normalized_salaries[i]:.3f}")

# ==================== 第4步：最终特征向量 ====================
print("\n==================== 最终特征向量 ====================")
print("每个候选人会被表示成一个向量：[是否博士, 是否本科, 是否硕士, 年龄(归一化), 经验(归一化), 期望工资(归一化)]")
print()

for i, c in enumerate(candidates):
    # 拼接 One-Hot + 归一化后的数值特征
    feature_vector = education_encoded[c["name"]] + [normalized_ages[i], normalized_experiences[i], normalized_salaries[i]]
    print(f"{c['name']}: {[round(x, 3) for x in feature_vector]}")

print("\n✅ 特征工程完成！这些向量可以直接喂给机器学习模型了。")
