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
