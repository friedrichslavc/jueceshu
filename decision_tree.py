import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pydotplus
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv('data/data.csv')

# 数据预处理
print(f"数据集形状: {data.shape}")
print("\n数据集前5行:")
print(data.head())

# 检查缺失值
print("\n缺失值统计:")
print(data.isnull().sum())

# 将分类特征转换为数值型
# 性别编码
data['sex'] = data['sex'].map({'Male': 0, 'Female': 1})

# 设备和操作系统编码
device_mapping = {'mobile': 0, 'desktop': 1}
data['device'] = data['device'].map(device_mapping)

os_mapping = {'android': 0, 'iOS': 1, 'windows': 2, 'mac': 3, 'other': 4}
data['operative_system'] = data['operative_system'].map(os_mapping)

# 来源编码
source_mapping = {'Direct': 0, 'Seo': 1, 'Ads': 2}
data['source'] = data['source'].map(source_mapping)

# 选择特征和目标变量
# 使用confirmation_page作为目标变量，表示用户是否完成了购买流程
features = ['new_user', 'age', 'sex', 'market', 'device', 'operative_system', 'source', 'total_pages_visited']
X = data[features]
y = data['confirmation_page']

# 查看目标变量分布
print("\n目标变量(confirmation_page)分布:")
print(y.value_counts())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练决策树模型
def train_decision_tree(max_depth=3, criterion='gini'):
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    clf.fit(X_train, y_train)
    
    # 评估模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 可视化决策树
    dot_data = export_graphviz(clf, out_file=None, 
                              feature_names=X.columns,  
                              class_names=['No', 'Yes'],  # 根据实际类别修改
                              filled=True, rounded=True,  
                              special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(f'tree_maxdepth{max_depth}_{criterion}.png')
    
    return clf, accuracy, cm

# 尝试不同参数
for max_depth in [3, 5]:
    for criterion in ['gini', 'entropy']:
        clf, accuracy, cm = train_decision_tree(max_depth, criterion)
        print(f'Max Depth: {max_depth}, Criterion: {criterion}, Accuracy: {accuracy:.2f}')
        print('Confusion Matrix:')
        print(cm)