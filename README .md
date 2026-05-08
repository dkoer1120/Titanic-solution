# Kaggle Titanic - 顶级生存率预测方案 (Top 10%)

本项目包含了一个在 Kaggle 经典比赛 [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) 中可达到 **0.80+** 评分的极简且高效的解决方案。

## 核心亮点 (Features)
本方案突破 0.80 分数瓶颈的核心在于**精细化的特征工程**与**模型融合**：
* **群体生存率 (Family Survival)**：通过提取 `Surname`（姓氏）和 `Ticket`（船票号），跨越训练集与测试集挖掘同伴/家属的生存情况（目标穿越的合法利用）。
* **甲板号提取 (Deck)**：即使 `Cabin` 缺失率极高，通过提取首字母计算物理距离，配合其他特征依然能提供决胜的判别力。
* **智能填补与分箱**：按照头衔 (`Title`) 和客舱等级 (`Pclass`) 智能填补缺失年龄。并使用 Pandas 转化为纯数值型分箱，完美适配 XGBoost。
* **三巨头软投票 (Soft Voting)**：融合了 `RandomForest`、`GradientBoosting` 和 `XGBoost`，并赋予不同的权重（1:2:2）以提升最终的泛化能力。

## 环境要求 (Requirements)
建议使用 Python 3.8+。安装依赖库：
```bash
pip install pandas numpy scikit-learn xgboost