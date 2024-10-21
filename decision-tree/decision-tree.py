# %%
import csv
import math
import random
import numpy as np
import pandas as pd

train_df = pd.read_csv('/Users/rachelloo/Downloads/Y4S1/CS3244/CS3244_Project/data/processed/train.csv',sep=",")
test_df = pd.read_csv('/Users/rachelloo/Downloads/Y4S1/CS3244/CS3244_Project/data/processed/test.csv',sep=",")
features = pd.read_csv('/Users/rachelloo/Downloads/Y4S1/CS3244/CS3244_Project/data/features.txt',sep=",")
# %%
X_train = train_df.drop(columns =['id','label'])
y_train  = train_df['label']
X_test = test_df.drop(columns =['id','label'])
y_test = test_df['label']
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# %%
param_dist = {
    'criterion': ['gini', 'entropy'],          # Criterion for splitting
    'max_depth': [None, 10, 20, 30, 40, 50],   # Depth of the tree
    'min_samples_split': np.arange(2, 10),     # Minimum number of samples required to split a node
    'min_samples_leaf': np.arange(1, 10),      # Minimum number of samples required at a leaf node
    'max_features': [None, 'sqrt', 'log2'],    # Number of features to consider for best split
}

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

random_search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_dist,
    n_iter=100,  # Number of different combinations to try
    scoring='accuracy',  # You can use other metrics like 'f1', 'roc_auc', etc.
    cv=10,  # 10-fold cross-validation
    random_state=42,
    n_jobs=-1  # Use all available cores for parallel processing
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Get the best parameters
print("Best hyperparameters found: ", random_search.best_params_)

# Predict using the best model
y_pred = random_search.best_estimator_.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
# %%
# Results
# Best hyperparameters found:  {'min_samples_split': np.int64(6), 'min_samples_leaf': np.int64(8), 'max_features': None, 'max_depth': 10, 'criterion': 'entropy'}
#              precision    recall  f1-score   support
#
#           1       0.77      0.95      0.85       496
#           2       0.82      0.71      0.76       471
#           3       0.89      0.81      0.84       420
#           4       0.83      0.78      0.80       508
#           5       0.81      0.86      0.84       556
#           6       1.00      0.99      0.99       545
#           7       0.92      0.52      0.67        23
#           8       0.69      0.90      0.78        10
#           9       0.69      0.75      0.72        32
#          10       0.46      0.44      0.45        25
#          11       0.71      0.61      0.66        49
#          12       0.43      0.33      0.38        27

#    accuracy                           0.84      3162
#   macro avg       0.75      0.72      0.73      3162
#weighted avg       0.84      0.84      0.84      3162
