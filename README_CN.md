# Kaggle Titanic 生存预测 - 0.80 突破方案

本项目提供了一个在 Kaggle 经典比赛 [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) 中稳定达到 **0.80** 评分的解决方案。该方案专注于挖掘乘客间的社会关系规律，而非单纯依赖个体物理特征。

## 核心技术路线

本方案的成功在于以下三点精准的策略：

1. **群体生存信号 (Family & Group Survival)**
   这是本代码最核心的提分点。通过 `Surname`（姓氏）、`Fare`（票价）和 `Ticket`（船票号）的组合，将训练集与测试集中的家属和同伴关联起来。通过观察同组人员在训练集中的生死情况，为测试集中的个体提供极强的预测信号。

2. **精细化特征处理**
   - **智能年龄填补**：不使用全局平均值，而是根据乘客的头衔（Title）和客舱等级（Pclass）进行分组填补，确保“儿童优先”原则在模型中得到准确体现。
   - **纯数值化分箱**：对年龄和票价进行分箱处理，并统一转换为整数类型，解决了 XGBoost 对类别型（Category）数据的兼容性问题。

3. **模型集成 (Ensemble Learning)**
   通过 `VotingClassifier` 融合了随机森林、梯度提升树（GBC）和 XGBoost。为了提升泛化能力，我们根据交叉验证的表现为 GBC 和 XGBoost 分配了双倍权重。

## 运行环境
- Python 3.x
- Pandas, Numpy, Scikit-learn, XGBoost

## 快速开始
1. 准备数据集：确保 `train.csv` 和 `test.csv` 与脚本在同一目录。
2. 运行脚本：`python titanic_solution.py`
3. 提交结果：将生成的 `submission.csv` 提交至 Kaggle 计分。