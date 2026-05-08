# ==========================================
# Kaggle Titanic - Top 10% Solution (0.80+)
# 保持原汁原味的数据处理流水线
# ==========================================

import numpy as np 
import pandas as pd 
import os
import re
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

# 消除 XGBoost 运行时的多余警告日志
warnings.filterwarnings('ignore')

# 1. 加载数据
train_data = pd.read_csv('train.csv')  # 注意：在本地或 GitHub 运行，请确保路径正确
test_data = pd.read_csv("test.csv")
df = pd.concat([train_data, test_data], ignore_index=True)

# 2. 基础缺失值填补与头衔、姓氏提取
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle','Ms','Mme'], ['Miss','Miss','Mrs'])

df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df['Age'].median())

# 3. 核心特征：群体生存率 (Family_Survival)
df['Family_Survival'] = 0.5

for grp, grp_df in df.groupby(['Surname', 'Fare']):
    if len(grp_df) != 1:
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if smax == 1.0:
                df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif smin == 0.0:
                df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

for grp, grp_df in df.groupby('Ticket'):
    if len(grp_df) != 1:
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if smax == 1.0:
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif smin == 0.0:
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

# 4. 数据分箱与家庭规模
df['FareBin'] = pd.qcut(df['Fare'], 5, labels=False)
df['AgeBin'] = pd.cut(df['Age'].astype(int), 5, labels=False)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 5. 类别编码
encoder = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title']:
    df[col] = encoder.fit_transform(df[col])

# 6. 清理冗余特征并拆分数据集
drop_cols = ['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Surname', 'Age', 'Fare']
df = df.drop(columns=drop_cols)

train_processed = df[:len(train_data)]
test_processed = df[len(train_data):]

X_train = train_processed.drop(columns=['Survived'])
y_train = train_processed['Survived'].astype(int)
X_test = test_processed.drop(columns=['Survived'])

# 7. 模型训练 (直接使用你网格搜索跑出来的最强参数)
print("正在初始化并训练模型...")

best_rf = RandomForestClassifier(
    max_depth=5, 
    min_samples_leaf=3, 
    min_samples_split=2, 
    n_estimators=300,
    random_state=1120
)

best_gbc = GradientBoostingClassifier(
    learning_rate=0.01, 
    max_depth=3, 
    min_samples_leaf=1, 
    min_samples_split=2, 
    n_estimators=200,
    random_state=1120
)

# 【修改点】：在这里加入了静音参数 verbosity=0，完美消除 XGB 的红色日志
best_xgb = XGBClassifier(
    learning_rate=0.01, 
    max_depth=3, 
    n_estimators=100,
    random_state=1120,
    verbosity=0,               # 静音模式
    use_label_encoder=False,
    eval_metric='logloss'
)

# 8. 模型融合与预测
voting_clf = VotingClassifier(
    estimators=[('rf', best_rf), ('gbc', best_gbc), ('xgb', best_xgb)], 
    voting='soft',
    weights=[1, 2, 2]
)

voting_clf.fit(X_train, y_train)
final_predictions = voting_clf.predict(X_test)

# 9. 生成提交文件
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": final_predictions
})

submission.to_csv('submission.csv', index=False)
print('生成文件成功')