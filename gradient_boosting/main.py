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
# %%
#Set random seed
random_seed = 31
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
# %%
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
print(train.head())
print(test.head())
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

# %%
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
# Load the best hyperparameters for XGBClassifier
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)
    return config
CONFIG_PATH = "config/"
config = load_config(CONFIG_PATH, "config.yaml")
print(config)
# %%
# Train the final models with the best hyperparameters
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
xgb_tuned.fit(X_train, y_train)

# Print the scores on the training and validation set
print('XGBoost - Training set score: {:.4f}'.format(xgb_tuned.score(X_train, y_train)))
print('XGBoost - Test set score: {:.4f}'.format(xgb_tuned.score(X_test, y_test)))
# %%
# Predict on the train set
y_train_pred_xgb = xgb_tuned.predict(X_train)

# Predict on the test set
y_test_pred_xgb = xgb_tuned.predict(X_test)

# Compute the confusion matrix
cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)

print('XGBoost Confusion matrix\n\n', cm_xgb)

# Print classification metrics for train set
print('\nXGBoost Classification Report for Train Set\n')
print(classification_report(y_train, y_train_pred_xgb))

# Print classification metrics for test set
print('\nXGBoost Classification Report for Test Set\n')
print(classification_report(y_test, y_test_pred_xgb))
# %%
# Pre-standardisation
# XGBoost - Training set score: 0.9999
# XGBoost - Test set score: 0.9099
# XGBoost Confusion matrix

# XGBoost Confusion matrix

#  [[480   4  12   0   0   0   0   0   0   0   0   0]
#  [ 40 426   4   0   0   0   0   0   0   0   1   0]
#  [  8  39 373   0   0   0   0   0   0   0   0   0]
#  [  0   0   0 422  83   0   1   1   1   0   0   0]
#  [  0   0   0  42 514   0   0   0   0   0   0   0]
#  [  0   0   0   0   0 545   0   0   0   0   0   0]
#  [  0   2   0   1   1   0  18   0   0   0   1   0]
#  [  0   0   0   0   0   0   0  10   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  26   0   6   0]
#  [  0   0   0   0   0   0   0   1   0  17   0   7]
#  [  2   0   0   3   0   1   1   1   9   0  32   0]
#  [  1   0   0   0   0   0   0   1   0  10   1  14]]

# XGBoost Classification Report for Train Set

#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00      1226
#            1       1.00      1.00      1.00      1073
#            2       1.00      1.00      1.00       987
#            3       1.00      1.00      1.00      1293
#            4       1.00      1.00      1.00      1423
#            5       1.00      1.00      1.00      1413
#            6       1.00      1.00      1.00        47
#            7       1.00      1.00      1.00        23
#            8       1.00      1.00      1.00        75
#            9       1.00      1.00      1.00        60
#           10       1.00      1.00      1.00        90
#           11       1.00      0.98      0.99        57

#     accuracy                           1.00      7767
#    macro avg       1.00      1.00      1.00      7767
# weighted avg       1.00      1.00      1.00      7767


# XGBoost Classification Report for Test Set

#               precision    recall  f1-score   support

#            0       0.90      0.97      0.93       496
#            1       0.90      0.90      0.90       471
#            2       0.96      0.89      0.92       420
#            3       0.90      0.83      0.86       508
#            4       0.86      0.92      0.89       556
#            5       1.00      1.00      1.00       545
#            6       0.90      0.78      0.84        23
#            7       0.71      1.00      0.83        10
#            8       0.72      0.81      0.76        32
#            9       0.63      0.68      0.65        25
#           10       0.78      0.65      0.71        49
#           11       0.67      0.52      0.58        27

#     accuracy                           0.91      3162
#    macro avg       0.83      0.83      0.82      3162
# weighted avg       0.91      0.91      0.91      3162

# Post-standardisation
# XGBoost - Training set score: 1.0000
# XGBoost - Test set score: 0.9051

# XGBoost Confusion matrix

#  [[480   4  12   0   0   0   0   0   0   0   0   0]
#  [ 50 412   7   0   0   0   0   0   0   0   2   0]
#  [ 16  35 369   0   0   0   0   0   0   0   0   0]
#  [  0   0   0 423  81   0   1   2   1   0   0   0]
#  [  0   0   0  42 514   0   0   0   0   0   0   0]
#  [  0   0   0   0   0 545   0   0   0   0   0   0]
#  [  0   3   0   1   1   0  18   0   0   0   0   0]
#  [  0   0   0   0   0   0   0  10   0   0   0   0]
#  [  0   0   0   0   0   0   0   1  25   0   6   0]
#  [  0   0   0   0   0   0   0   1   0  17   0   7]
#  [  2   0   0   2   0   1   1   1   8   0  33   1]
#  [  1   0   0   0   0   0   0   1   0   8   1  16]]

# XGBoost Classification Report for Train Set

#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00      1226
#            1       1.00      1.00      1.00      1073
#            2       1.00      1.00      1.00       987
#            3       1.00      1.00      1.00      1293
#            4       1.00      1.00      1.00      1423
#            5       1.00      1.00      1.00      1413
#            6       1.00      1.00      1.00        47
#            7       1.00      1.00      1.00        23
#            8       1.00      1.00      1.00        75
#            9       1.00      1.00      1.00        60
#           10       1.00      1.00      1.00        90
#           11       1.00      1.00      1.00        57

#     accuracy                           1.00      7767
#    macro avg       1.00      1.00      1.00      7767
# weighted avg       1.00      1.00      1.00      7767


# XGBoost Classification Report for Test Set

#               precision    recall  f1-score   support

#            0       0.87      0.97      0.92       496
#            1       0.91      0.87      0.89       471
#            2       0.95      0.88      0.91       420
#            3       0.90      0.83      0.87       508
#            4       0.86      0.92      0.89       556
#            5       1.00      1.00      1.00       545
#            6       0.90      0.78      0.84        23
#            7       0.62      1.00      0.77        10
#            8       0.74      0.78      0.76        32
#            9       0.68      0.68      0.68        25
#           10       0.79      0.67      0.73        49
#           11       0.67      0.59      0.63        27

#     accuracy                           0.91      3162
#    macro avg       0.82      0.83      0.82      3162
# weighted avg       0.91      0.91      0.90      3162

# %%
