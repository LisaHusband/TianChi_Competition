
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('testA.csv')


# 数据预处理
def preprocess_data(df):
    # 脱敏处理
    df = df.drop(columns=['postCode', 'n11', 'n12'])

    # 处理数值变量的缺失值
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # 处理分类变量的缺失值
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # 处理分类变量编码
    for column in cat_cols:
        df[column] = LabelEncoder().fit_transform(df[column])

    return df

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 特征选择和数据标准化
X = train_data.drop(columns=['id', 'isDefault'])
y = train_data['isDefault']
X_test = test_data.drop(columns=['id'])

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# 指定类别型特征的列名或索引
cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object' or 'category' in str(X_train[col].dtype)]

# 创建模型
cat_model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    cat_features=cat_cols,
    early_stopping_rounds=5000,
    # custom_metric=['AUC', 'F1', 'Recall', 'Precision'],
    # class_weights = [1, 4], # 非违约:违约 = 1:4（经验值，20%违约率适合此比例）
    verbose=100
)

# 拟合模型
cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
)

# 预测验证集概率（违约概率）
y_val_pred_cat = cat_model.predict_proba(X_val)[:, 1]

# 计算 AUC
auc_score_cat = roc_auc_score(y_val, y_val_pred_cat)
print(f'CatBoost Validation AUC Score: {auc_score_cat:.4f}')

best_model = cat_model
best_auc = auc_score_cat

print(f'Best Model is {best_model}')
print(f'Best Model AUC: {best_auc}')

# 使用全量数据训练最佳模型
best_model.fit(X_train, y_train)

# 在测试集上进行预测
test_pred = best_model.predict_proba(X_test)[:, 1]

# 结果提交
submission = pd.DataFrame({'id': test_data['id'], 'isDefault': test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")

