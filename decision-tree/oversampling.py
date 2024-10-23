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
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE

clf = DecisionTreeClassifier(random_state=31)
decision_tree_model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# %%
pipeline = Pipeline([
    ('smote', SMOTE(random_state=31)),               # SMOTE for handling class imbalance
    ('scaler', StandardScaler()),                    # Standardization
    ('model', decision_tree_model)                                     # Decision Tree model
])
# Perform cross-validation to evaluate the model
scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='f1_micro')  # Change scoring as needed

# Print the average F1 score from cross-validation
print("Average F1 Score (with SMOTE and Standardization):", np.mean(scores))
# Average F1 Score (with SMOTE and Standardization): 0.8126885722246546

# Fit the pipeline on the training data and evaluate on the test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("F1 Score on Test Set:", f1_score(y_test, y_pred, average='micro'))
# F1 Score on Test Set: 0.8314358001265022
# %%
