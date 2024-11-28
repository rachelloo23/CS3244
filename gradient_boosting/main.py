# %%
import numpy as np
import xgboost as xgb
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import yaml
from imblearn.over_sampling import SMOTE
from collections import Counter
from module import *
# %%
#Set random seed
random_seed = 31
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
# %%
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
train_8 = pd.read_csv("../data/processed/train_8.csv")
test_8 = pd.read_csv("../data/processed/test_8.csv")
train_9 = pd.read_csv("../data/processed/train_9.csv")
test_9 = pd.read_csv("../data/processed/test_9.csv")

print(train.head())
print(test.head())
print(train_8.head())
print(test_8.head())
# %%
train = train.drop(["id"], axis=1)
test = test.drop(["id"], axis=1)
print(train.head())
print(test.head())
# %%
# Split the data into X (features) and y (labels)
X_train = train.drop(["label"], axis=1)
y_train = train["label"]
y_train = y_train - 1
y_test = test["label"] - 1
X_test = test.drop(["label"], axis=1)

X_train_8 = train_8.drop(["label"], axis=1)
y_train_8 = train_8["label"]
y_train_8 = y_train_8 - 1
y_test_8 = test_8["label"] - 1
X_test_8 = test_8.drop(["label"], axis=1)

X_train_9 = train_9.drop(["label"], axis=1)
y_train_9 = train_9["label"]
y_train_9 = y_train_9 - 1
y_test_9 = test_9["label"] - 1
X_test_9 = test_9.drop(["label"], axis=1)
# # %%
# # Standardize the features
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)
# # X_train_8 = scaler.fit_transform(X_train_8)
# # X_test_8 = scaler.transform(X_test_8)
# # X_train_9 = scaler.fit_transform(X_train_9)
# # X_test_9 = scaler.transform(X_test_9)
# # %%
# # 1. Default params on original data
# # 2. Default params on feature selected data
# # 3. Default params on oversampled original data
# # 4. Default params on oversampled feature selected data
# # 5. Tune on the best score from 1, 2, 3, 4.
# # %%
# # 1. Default params on original data
# xgb = XGBClassifier(
#     objective="multi:softmax",
#     random_state=random_seed,
#     eval_metric="mlogloss"
# )
# xgb.fit(X_train, y_train)
# filename = "./results/defOrig.txt"
# writeResults(xgb, X_train, y_train, X_test, y_test, filename)
# # %%
# # 2. Default params on feature selected data
# scaler = StandardScaler()
# X_train_8 = scaler.fit_transform(X_train_8)
# X_test_8 = scaler.transform(X_test_8)

# xgb = XGBClassifier(
#     objective="multi:softmax",
#     random_state=random_seed,
#     eval_metric="mlogloss"
# )
# xgb.fit(X_train_8, y_train_8)
# filename = "./results/defFeat_8.txt"
# writeResults(xgb, X_train_8, y_train_8, X_test_8, y_test_8, filename)
# xgb = XGBClassifier(
#     objective="multi:softmax",
#     random_state=random_seed,
#     eval_metric="mlogloss"
# )
# scaler = StandardScaler()
# X_train_9 = scaler.fit_transform(X_train_9)
# X_test_9 = scaler.transform(X_test_9)
# xgb.fit(X_train_9, y_train_9)
# filename = "./results/defFeat_9.txt"
# writeResults(xgb, X_train_9, y_train_9, X_test_9, y_test_9, filename)
# # %%
# # 3. Default params on oversampled original data
# smote = SMOTE(random_state=random_seed)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# scaler = StandardScaler()
# X_train_smote = scaler.fit_transform(X_train_smote)
# X_test = scaler.transform(X_test)
# xgb = XGBClassifier(
#     objective="multi:softmax",
#     random_state=random_seed,
#     eval_metric="mlogloss"
# )
# xgb.fit(X_train_smote, y_train_smote)
# filename = "./results/defOrigSmote.txt"
# writeResults(xgb, X_train_smote, y_train_smote, X_test, y_test, filename)
# # %%
# # 4. Default params on oversampled feature selected data
# smote = SMOTE(random_state=random_seed)
# X_train_smote_8, y_train_smote_8 = smote.fit_resample(X_train_8, y_train_8)
# scaler = StandardScaler()
# X_train_smote_8 = scaler.fit_transform(X_train_smote_8)
# X_test_8 = scaler.transform(X_test_8)
# xgb = XGBClassifier(
#     objective="multi:softmax",
#     random_state=random_seed,
#     eval_metric="mlogloss"
# )
# xgb.fit(X_train_smote_8, y_train_smote_8)
# filename = "./results/defFeatSmote_8.txt"
# writeResults(xgb, X_train_smote_8, y_train_smote_8, X_test_8, y_test_8, filename)
# X_train_smote_9, y_train_smote_9 = smote.fit_resample(X_train_9, y_train_9)
# scaler = StandardScaler()
# X_train_smote_9 = scaler.fit_transform(X_train_smote_9)
# X_test_9 = scaler.transform(X_test_9)
# xgb = XGBClassifier(
#     objective="multi:softmax",
#     random_state=random_seed,
#     eval_metric="mlogloss"
# )
# xgb.fit(X_train_smote_9, y_train_smote_9)
# filename = "./results/defFeatSmote_9.txt"
# writeResults(xgb, X_train_smote_9, y_train_smote_9, X_test_9, y_test_9, filename)
# %%
# Load the best hyperparameters for XGBClassifier
CONFIG_PATH = "config/"
config = load_config(CONFIG_PATH, "config.yaml")
print(config)
# %%
# Train the final models with the best hyperparameters
smote = SMOTE(random_state=random_seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)
xgb_tuned = XGBClassifier(
    n_estimators=config['n_estimators'],
    max_depth=config['max_depth'],
    learning_rate=config['learning_rate'],
    subsample=config['subsample'],
    colsample_bytree=config['colsample_bytree'],
    gamma=config['gamma'],
    reg_alpha=config['reg_alpha'],
    reg_lambda=config['reg_lambda'],
    random_state=random_seed
)
xgb_tuned.fit(X_train_smote, y_train_smote)
filename = "./results/tuned_Smote.txt"
writeResults(xgb_tuned, X_train_smote, y_train_smote, X_test, y_test, filename)
# %%
if __name__ == "__main__":
    main()
