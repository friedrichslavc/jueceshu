# 决策树建模实验项目

## 项目结构
```
./
├── data/               # 数据目录
│   └── data.csv       # 实验数据集
├── docs/              # 文档目录
│   ├── 第三次实验题目.md  # 实验指导文档
│   └── 京东消费者基础信息与浏览行为数据集_readme.md
├── decision_tree.py   # 决策树模型实现
└── README.md          # 项目说明文件
```

## 实验步骤
1. 准备数据
   - 确保data.csv文件包含正确的特征和目标列
   - 数据应包含数值型和类别型特征

2. 运行决策树模型
   ```bash
   python decision_tree.py
   ```
   - 程序将输出不同参数组合下的准确率
   - 生成决策树可视化图片(tree_maxdepthX_criterionY.png)

3. 分析结果
   - 比较不同max_depth和criterion对模型性能的影响
   - 观察决策树结构，理解关键分裂点

## 依赖安装
```bash
pip install scikit-learn pandas numpy matplotlib pydotplus graphviz
```

## 注意事项
- 确保已安装Graphviz并配置环境变量
- 根据实际数据修改decision_tree.py中的目标列名
- 实验报告应包含模型评估结果和决策树分析