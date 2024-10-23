# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import random
# %%
subject_id = pd.read_csv("../data/Train/subject_id_train.txt", header=None)
X_train =  pd.read_csv("../data/Train/X_train.txt", header=None, delim_whitespace=True)
y_train = pd.read_csv("../data/Train/y_train.txt", header=None)
X_test = pd.read_csv("../data/Test/X_test.txt", header=None, delim_whitespace=True)
y_test = pd.read_csv("../data/Test/y_test.txt", header=None)
features = pd.read_csv("../data/features.txt", header=None)
# %%
def create_df(X_train, subject_id, y_train, features):
    # Create a copy of the X_train dataframe
    train = X_train.copy()

    # Add "id" and "label" columns
    train["id"] = subject_id.iloc[:, 0]
    train["label"] = y_train.iloc[:, 0]

    # Use feature names from 'features'
    feature_names = features.iloc[:, 0].values
    train.columns = np.append(feature_names, ["id", "label"])

    # Display the created dataframe
    display(train)

    # Display the value counts of 'label' where id == 1
    display(train[train["id"] == 1]["label"].value_counts())

    return train
train = create_df(X_train, subject_id, y_train, features)
test = create_df(X_test, subject_id, y_test, features)
# %%
train.to_csv("../data/processed/train.csv", index=False)
test.to_csv("../data/processed/test.csv", index=False)
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

clf = DecisionTreeClassifier(random_state=31)
decision_tree_model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 score:", f1)
# f1 score: 0.7101925262849368
# %%
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', decision_tree_model )])
# 1. Using cross_val_score
scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring=make_scorer(f1_score, average='micro'))

print("Base model scores: ", scores)
# Base model scores:  [0.88803089 0.78120978 0.87902188 0.71943372 0.83912484 0.78378378
# 0.86229086 0.8814433  0.89046392 0.8685567 ]

print("Average f1 score: ", scores.mean())
#Average f1 score:  0.8393359670421526
# %%
# 2. Using RandomizedSearchCV
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', decision_tree_model )])
# Define hyperparameter space for RandomizedSearchCV
param_dist = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': np.arange(2, 10),
    'classifier__min_samples_leaf': np.arange(1, 10),
    'classifier__max_features': [None, 'sqrt', 'log2'],
}
# Define RandomizedSearchCV with 10-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings that are sampled
    scoring=make_scorer(f1_score, average='micro'),
    cv=10,  # 10-fold cross-validation for each parameter setting
    random_state=31,
    n_jobs=-1  # Use all available cores
)
# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)
# Get the best parameters
print("Best hyperparameters found: ", random_search.best_params_)
# Predict using the best model
y_pred = random_search.best_estimator_.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# %%
# Results
#Best hyperparameters found:  {'classifier__min_samples_split': np.int64(6), 'classifier__min_samples_leaf': np.int64(6), 'classifier__max_features': None, 'classifier__max_depth': None, 'classifier__criterion': 'entropy'}
#              precision    recall  f1-score   support
#
#           1       0.79      0.91      0.85       496
#           2       0.81      0.67      0.73       471
#           3       0.81      0.83      0.82       420
#           4       0.81      0.79      0.80       508
#           5       0.81      0.85      0.83       556
#           6       1.00      0.99      0.99       545
#           7       0.75      0.52      0.62        23
#           8       0.69      0.90      0.78        10
#           9       0.64      0.72      0.68        32
#          10       0.52      0.44      0.48        25
#          11       0.64      0.59      0.62        49
#          12       0.50      0.44      0.47        27
#
#    accuracy                           0.83      3162
#   macro avg       0.73      0.72      0.72      3162
#weighted avg       0.83      0.83      0.83      3162

