# %%
import csv
import math
import random
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

random_seed = 31

# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]
#%%
# Use SMOTE (can use cause all data is numerical)
# Import necessary libraries
from imblearn.over_sampling import SMOTE
from collections import Counter

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=random_seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified k-fold to ensure each fold has a similar class distribution
strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

# Lists to store cross-validation F1 scores for training and test
train_f1_scores = []
test_f1_scores = []

# Loop through different k values
k_range = range(1, 20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    fold_train_scores = []
    fold_test_scores = []
    
    # Perform stratified k-fold split
    for train_index, test_index in strat_kfold.split(X_train, y_train):
        # Use iloc to select the correct indices
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        # Standardize the data for each fold
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Fit the model on the training fold
        knn.fit(X_train_fold, y_train_fold)
        
        # Predict on training and test folds
        y_train_pred = knn.predict(X_train_fold)
        y_test_pred = knn.predict(X_test_fold)
        
        # Compute F1 scores for training and test sets
        train_f1 = f1_score(y_train_fold, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test_fold, y_test_pred, average='weighted')
        
        fold_train_scores.append(train_f1)
        fold_test_scores.append(test_f1)
    
    # Store the average F1 scores for this k
    train_f1_scores.append(np.mean(fold_train_scores))
    test_f1_scores.append(np.mean(fold_test_scores))

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'Training F1 Score (weighted)': train_f1_scores,
    'Test F1 Score (weighted)': test_f1_scores
})

# Determine the best k (k with the highest Test Weighted F1)
best_k = k_range[test_f1_scores.index(max(test_f1_scores))]

print(f"Best k: {best_k}")
print(results_df_f1)
     k  Training F1 Score (weighted)  Test F1 Score (weighted)
0    1                      1.000000                  0.955996
1    2                      0.974970                  0.933142
2    3                      0.984280                  0.951558
3    4                      0.972547                  0.945232
4    5                      0.971397                  0.949761
5    6                      0.967499                  0.945020
6    7                      0.966219                  0.947441
7    8                      0.963270                  0.944970
8    9                      0.961014                  0.941242
9   10                      0.959214                  0.942171
10  11                      0.957141                  0.940949
11  12                      0.957083                  0.943139
12  13                      0.954363                  0.940155
13  14                      0.954072                  0.939827
14  15                      0.951147                  0.937553
15  16                      0.951489                  0.938665
16  17                      0.948712                  0.936501
17  18                      0.949056                  0.936849
18  19                      0.946702                  0.935046
