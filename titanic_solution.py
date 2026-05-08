import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

# 忽略警告日志
warnings.filterwarnings('ignore')

def load_and_preprocess(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    df = pd.concat([train_data, test_data], ignore_index=True)

    # 1. 填补缺失值
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 2. 提取头衔 (Title)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle','Ms','Mme'], ['Miss','Miss','Mrs'])

    # 3. 提取姓氏 (Surname)
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

    # 4. 精准填补年龄 (Age)
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 5. 提取甲板号 (Deck) - 冲刺 0.80 的关键
    df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'U')
    df['Deck'] = df['Deck'].replace(['T'], 'U')

    # 6. 群体生存率 (Family_Survival)
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

    # 7. 数据分箱 (转化为整数，完美适配 XGBoost)
    df['FareBin'] = pd.qcut(df['Fare'], 5, labels=False)
    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5, labels=False)

    # 8. 家庭规模
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 9. 类别特征编码
    encoder = LabelEncoder()
    for col in ['Sex', 'Embarked', 'Title', 'Deck']:
        df[col] = encoder.fit_transform(df[col])

    # 10. 清理特征
    drop_cols = ['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Surname', 'Age', 'Fare']
    df = df.drop(columns=drop_cols)

    # 拆分数据集
    train_processed = df[:len(train_data)]
    test_processed = df[len(train_data):]

    X_train = train_processed.drop(columns=['Survived'])
    y_train = train_processed['Survived'].astype(int)
    X_test = test_processed.drop(columns=['Survived'])

    return X_train, y_train, X_test, test_data["PassengerId"]

def train_and_predict(X_train, y_train, X_test):
    # 模型 1: 随机森林
    rf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=5, 
        min_samples_split=2, 
        min_samples_leaf=3, 
        random_state=1120
    )
    
    # 模型 2: 梯度提升树
    gbc = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.01, 
        max_depth=3, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=1120
    )

    # 模型 3: XGBoost (已消除烦人的 Warning)
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        random_state=1120,
        verbosity=0,               # 静音模式
        use_label_encoder=False,
        eval_metric='logloss'      
    )

    # 软投票加权融合 (Weighted Soft Voting)
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('gbc', gbc), ('xgb', xgb)],
        voting='soft',
        weights=[1, 2, 2]
    )

    print("正在训练融合模型...")
    voting_clf.fit(X_train, y_train)

    print("正在生成预测结果...")
    return voting_clf.predict(X_test)

if __name__ == "__main__":
    # 请确保 train.csv 和 test.csv 与此代码在同一目录下
    TRAIN_PATH = 'train.csv'
    TEST_PATH = 'test.csv'

    print("开始加载并预处理数据...")
    try:
        X_train, y_train, X_test, passenger_ids = load_and_preprocess(TRAIN_PATH, TEST_PATH)
        
        predictions = train_and_predict(X_train, y_train, X_test)

        submission = pd.DataFrame({
            "PassengerId": passenger_ids,
            "Survived": predictions
        })

        submission.to_csv('submission.csv', index=False)
        print("大功告成！已成功生成 submission.csv")
        
    except FileNotFoundError:
        print(f"错误: 找不到 {TRAIN_PATH} 或 {TEST_PATH}，请确保数据集文件已下载并放在同级目录。")