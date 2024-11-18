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
# %%
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_8 = scaler.fit_transform(X_train_8)
X_test_8 = scaler.transform(X_test_8)
X_train_9 = scaler.fit_transform(X_train_9)
X_test_9 = scaler.transform(X_test_9)

# %%
def writeResults(model, X_train, y_train, X_test, y_test, filename):
    """
    Evaluate the model and save the confusion matrix and classification reports to a text file.

    Parameters:
    - model: Trained model to evaluate.
    - X_train: Training feature set.
    - y_train: Training labels.
    - X_test: Test feature set.
    - y_test: Test labels.
    - filename: Name of the output text file (string).
    """
    # Evaluate and print training and test set scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('Model - Training set score: {:.4f}'.format(train_score))
    print('Model - Test set score: {:.4f}'.format(test_score))

    # Predict on the training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute confusion matrix for the test set
    cm = confusion_matrix(y_test, y_test_pred)

    # Generate classification reports
    report_train = classification_report(y_train, y_train_pred)
    report_test = classification_report(y_test, y_test_pred)

    # Save confusion matrix and classification reports to a text file
    with open(filename, 'w') as f:
        # Write confusion matrix
        f.write('Confusion Matrix for Test Set:\n')
        f.write(str(cm))
        f.write('\n\n')

        # Write classification report for the training set
        f.write('Classification Report for Training Set:\n')
        f.write(report_train)
        f.write('\n\n')

        # Write classification report for the test set
        f.write('Classification Report for Test Set:\n')
        f.write(report_test)

    # Indicate that the results have been saved
    print(f'Results have been saved to {filename}')
# %%
# 1. Default params on original data
# 2. Default params on feature selected data
# 3. Default params on oversampled original data
# 4. Default params on oversampled feature selected data
# 5. Tune on the best score from 1, 2, 3, 4.
# %%
# 1. Default params on original data
xgb = XGBClassifier(
    objective="multi:softmax",
    random_state=random_seed,
    eval_metric="mlogloss"
)
xgb.fit(X_train, y_train)
filename = "./results/defOrig.txt"
writeResults(xgb, X_train, y_train, X_test, y_test, filename)
# %%
# 2. Default params on feature selected data
xgb.fit(X_train_8, y_train_8)
filename = "./results/defFeat_8.txt"
writeResults(xgb, X_train_8, y_train_8, X_test_8, y_test_8, filename)
xgb.fit(X_train_9, y_train_9)
filename = "./results/defFeat_9.txt"
writeResults(xgb, X_train_9, y_train_9, X_test_9, y_test_9, filename)
# %%
# 3. Default params on oversampled original data
smote = SMOTE(random_state=random_seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
xgb.fit(X_train_smote, y_train_smote)
filename = "./results/defOrigSmote.txt"
writeResults(xgb, X_train_smote, y_train_smote, X_test, y_test, filename)
# %%
# 4. Default params on oversampled feature selected data
smote = SMOTE(random_state=random_seed)
X_train_smote_8, y_train_smote_8 = smote.fit_resample(X_train_8, y_train_8)
xgb.fit(X_train_smote_8, y_train_smote_8)
filename = "./results/defFeatSmote_8.txt"
writeResults(xgb, X_train_smote_8, y_train_smote_8, X_test_8, y_test_8, filename)
X_train_smote_9, y_train_smote_9 = smote.fit_resample(X_train_9, y_train_9)
xgb.fit(X_train_smote_9, y_train_smote_9)
filename = "./results/defFeatSmote_9.txt"
writeResults(xgb, X_train_smote_9, y_train_smote_9, X_test_9, y_test_9, filename)
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
smote = SMOTE(random_state=random_seed)
X_train_smote_8, y_train_smote_8 = smote.fit_resample(X_train_8, y_train_8)
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
xgb_tuned.fit(X_train_smote_8, y_train_smote_8)
filename = "./results/tuned_FeatSmote_8.txt"
writeResults(xgb_tuned, X_train_smote_8, y_train_smote_8, X_test_8, y_test_8, filename)
# %%
