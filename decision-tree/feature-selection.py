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
# Calculate the correlation matrix for the training set
correlation_matrix = X_train.corr().abs()
# Create a mask for the upper triangle of the correlation matrix
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
# Find the index of feature columns that have a correlation greater than 0.8
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
# Drop the features from both the training and testing set
X_train_selected = X_train.drop(columns=to_drop)
X_test_selected = X_test.drop(columns=to_drop)
# Display the results
print("Dropped features:", to_drop)
print("Original Dataframe shape: ", X_train.shape)
print("Reduced DataFrame shape:", X_train_selected.shape)

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

clf = DecisionTreeClassifier(random_state=31)
decision_tree_model = clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 score:", f1)
# f1 score: 0.7290634944622226
# %%
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', decision_tree_model )])
# 1. Using cross_val_score
scores = cross_val_score(pipeline, X_train_selected, y_train, cv=10, scoring=make_scorer(f1_score, average='micro'))

print("Base model scores: ", scores)
# Base model scores:  [0.86357786 0.76962677 0.81338481 0.76576577 0.84813385 0.80952381
# 0.85585586 0.90850515 0.88530928 0.8556701 ]

print("Average f1 score: ", scores.mean())
#Average f1 score:  0.8375353261951201
# %%
# 2. Using RandomizedSearchCV
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', decision_tree_model)])
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
random_search.fit(X_train_selected, y_train)
# Get the best parameters
print("Best hyperparameters found: ", random_search.best_params_)
# Predict using the best model
y_pred = random_search.best_estimator_.predict(X_test_selected)
# %%
from sklearn.metrics import classification_report
print(classification_report(y_train, random_search.best_estimator_.predict(X_train_selected) , digits=5))
print(classification_report(y_test, y_pred, digits=5))
# Best hyperparameters found:  {'classifier__min_samples_split': np.int64(5), 'classifier__min_samples_leaf': np.int64(9), 'classifier__max_features': None, 'classifier__max_depth': 40, 'classifier__criterion': 'entropy'}
#              precision    recall  f1-score   support
#
#           1       0.84      0.93      0.88       496
#           2       0.83      0.81      0.82       471
#           3       0.85      0.79      0.82       420
#           4       0.83      0.80      0.81       508
#           5       0.82      0.88      0.85       556
#           6       1.00      1.00      1.00       545
#           7       0.50      0.70      0.58        23
#           8       0.75      0.30      0.43        10
#           9       0.47      0.53      0.50        32
#          10       0.47      0.32      0.38        25
#          11       0.65      0.27      0.38        49
#          12       0.50      0.59      0.54        27
#
#    accuracy                           0.85      3162
#   macro avg       0.71      0.66      0.67      3162
#weighted avg       0.85      0.85      0.85      3162
# %%
misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test.values.ravel(), y_pred)) if true != pred]
print("Misclassified instances:")
for i in misclassified_indices:
    print(f"Index: {i}, True label: {y_test.iloc[i, 0]}, Predicted label: {y_pred[i]}, Features: {X_test.iloc[i].values}")

print("Misclassified labels and index:")
for i in misclassified_indices:
    print(f"Index: {i}, True label: {y_test.iloc[i, 0]}, Predicted label: {y_pred[i]}")
#Misclassified instances:
#Index: 13, True label: 5, Predicted label: 4, Features: [ 0.03570833 -0.01327702 -0.02055589 -0.99698625 -0.99067633 -0.99166519
# -0.99739269 -0.991612   -0.99324678 -0.80088366 -0.75537678 -0.71577542
#  0.83516416  0.6961986   0.66923863 -0.991762   -0.99997119 -0.99976327
# -0.99991594 -0.99702084 -0.99387578 -0.99365765 -0.88373493 -0.95034677
# -0.62757074  0.43730913 -0.24972741  0.26422063 -0.43862627  0.05804684
# -0.09057449  0.13833332 -0.01153947  0.49535705 -0.26459948  0.12206139
#  0.1994329   0.39889068  0.09050956  0.14431142  0.91983955 -0.31433982
#  0.12916605 -0.99422291 -0.9841596  -0.99577134 -0.99444199 -0.98479805
# -0.99626316  0.84993008 -0.33507065  0.12178876  0.94071289 -0.29327567
#  0.13002227  0.07127618  0.78746973 -0.82391987 -0.96990246 -0.99524098
# -0.98705492 -0.99809306 -0.93170162 -1.         -0.91257575 -0.26981917
#  0.27972083 -0.29027056  0.30145549 -0.03034419  0.04560765 -0.06313135
#  0.08133712 -0.44324844  0.46342426 -0.48374158  0.50334883  0.99279759
#  0.62429382  0.7135155   0.07917393  0.01849573 -0.02170381 -0.99361731
# -0.98564396 -0.99061427 -0.99343543 -0.98352596 -0.98928107 -0.99081662
# -0.99290962 -0.99103895  0.99551573  0.98264569  0.99255188 -0.99128992
# -0.99993614 -0.9997592  -0.99981728 -0.99173852 -0.9798149  -0.9862259
# -0.76690966 -0.68048265 -0.70593407  0.54858194  0.03258288  0.6597768
# 0.44242367  0.11121481  0.00807146  0.1613651   0.33455164  0.23461794
#  0.01659009  0.06927747  0.41146424  0.12555833  0.39768376 -0.0830022
# -0.02694173 -0.01717211 -0.07548552 -0.99388721 -0.98818963 -0.98593764
# -0.9945141  -0.98973157 -0.98685478 -0.90680874 -0.95116417 -0.75812223
#  0.84808895  0.88753192  0.82019171 -0.98899218 -0.99997238 -0.99989508
# -0.99984717 -0.99509182 -0.99057375 -0.9898564  -0.47782302 -0.30252804
#...
#Index: 3113, True label: 3, Predicted label: 1
#Index: 3114, True label: 2, Predicted label: 3
#Index: 3154, True label: 2, Predicted label: 1
#Index: 3160, True label: 2, Predicted label: 1

# %%
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=decision_tree_model, cv=5, scoring='f1_weighted')  # Adjust scoring metric as needed
rfecv.fit(X_train, y_train)

print("Optimal number of features:", rfecv.n_features_)
print("Selected features:", X_train.columns[rfecv.support_])
#Optimal number of features: 21
#Selected features: Index([  1,   3,   9,  37,  49,  50,  51,  52,  53,  57,  69, 139, 166, 197,
#       209, 274, 418, 448, 451, 503, 559],
#      dtype='int64')
# %%
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(rfecv.cv_results_['mean_test_score']) + 1),  # X-axis: Number of features
    rfecv.cv_results_['mean_test_score'],                    # Y-axis: Mean F1 score
    marker='o',
    label='Mean F1 Score'
)
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross-Validated F1 Score")
plt.title("RFECV - Feature Selection with F1 Scoring")
plt.axvline(rfecv.n_features_, color="red", linestyle="--", label="Optimal Features")
plt.legend()
plt.grid()
plt.show()
# %%
print(f"Optimal number of features: {rfecv.n_features_}")
selected_features = np.array(features)[rfecv.support_]  # Assuming 'feature_names' is a list of all feature names
print(f"Selected features: {selected_features}")
# %%
