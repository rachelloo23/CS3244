#%%
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np

# Define parameter space with limited values
param_dist = {
    'C': [0.1, 1, 10],            # Limit the values of C
    'gamma': ['scale', 'auto'],    # Restrict gamma values
    'kernel': ['linear', 'rbf']    # Use just two kernels
}

# Reduce the size of the dataset (use 50% of training data for tuning)
X_train_sub = X_train.sample(frac=0.5, random_state=42)
y_train_sub = y_train.loc[X_train_sub.index]

# StratifiedKFold with fewer splits (reduce from 5 to 3 for efficiency)
strat_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# RandomizedSearchCV with limited number of iterations and parallel processing
random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_dist,
    n_iter=10,                         # Reduce the number of iterations
    scoring='f1_weighted',              # Optimize for f1_weighted score
    cv=strat_kfold,                     # Use 3-fold cross-validation
    verbose=1,                          # Set verbosity to see progress
    n_jobs=2                            # Limit parallel jobs to 2
)

# Fit the model with the subset of the data
random_search.fit(X_train_sub, y_train_sub)

# Print best parameters found
print(f"Best Parameters: {random_search.best_params_}")

# Best Parameters: {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}

# Evaluate best model on the full training and test data (after tuning)
best_model = random_search.best_estimator_

# Fit on the entire training set
best_model.fit(X_train, y_train)

# Predictions on train and test data
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Print classification reports for train and test
from sklearn.metrics import classification_report, f1_score

print("Training Set Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))

# Compare f1_weighted on training and test sets
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

print(f"Training f1_weighted: {train_f1_weighted:.2f}")
print(f"Test f1_weighted: {test_f1_weighted:.2f}")
