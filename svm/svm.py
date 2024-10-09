#%%
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]

#%%
svc = SVC()
svc.fit(X_train, y_train)
# print(accuracy_score(y_test, svc.predict(X_test))) # 0.9367488931056294
# print(accuracy_score(y_train, svc.predict(X_train))) # 0.9716750354062057

# print(f1_score(y_test, svc.predict(X_test), average='weighted')) # 0.9360546876940178
# print(f1_score(y_train, svc.predict(X_train), average='weighted')) # 0.971551119209504

#%%
# Tune model parameters
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import numpy as np

# Define the parameter grid with a reduced search space
param_distributions = {
    'C': [0.1, 1, 10, 100],  # Limited C values
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Limited gamma values
    'kernel': ['linear', 'rbf']  # Only rbf and linear kernels
}

# Set up the SVC model
svc = SVC()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    svc,
    param_distributions=param_distributions,
    n_iter=10,  
    scoring='f1_weighted',  # Use weighted F1 score as evaluation metric
    cv=3,  # 3-fold cross-validation to balance computation and evaluation
    verbose=1,  # Print search progress
    random_state=31,  # Ensure reproducibility
    n_jobs=-1  # Use all available cores
)

# Perform the random search on the training set
random_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best parameters found:", random_search.best_params_)

# Evaluate the best model on the test set
y_test_pred = random_search.best_estimator_.predict(X_test)
y_train_pred = random_search.best_estimator_.predict(X_train)

# Print the final classification report for the test set
print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))

train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

print(f"Training f1_weighted: {train_f1_weighted:.2f}")
print(f"Test f1_weighted: {test_f1_weighted:.2f}")
# Best parameters found: {'kernel': 'rbf', 'gamma': 'scale', 'C': 100}
# Test Set Classification Report:
#               precision    recall  f1-score   support

#            1       0.96      0.99      0.98       496
#            2       0.95      0.97      0.96       471
#            3       0.99      0.95      0.97       420
#            4       0.97      0.90      0.93       508
#            5       0.92      0.98      0.95       556
#            6       1.00      1.00      1.00       545
#            7       0.95      0.83      0.88        23
#            8       0.91      1.00      0.95        10
#            9       0.70      0.66      0.68        32
#           10       0.73      0.76      0.75        25
#           11       0.71      0.73      0.72        49
#           12       0.74      0.63      0.68        27

#     accuracy                           0.95      3162
#    macro avg       0.88      0.87      0.87      3162
# weighted avg       0.95      0.95      0.95      3162

# Training f1_weighted: 1.00
# Test f1_weighted: 0.95