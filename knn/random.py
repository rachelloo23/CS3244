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

# First part: compare with feature selection and without
# Second part: oversample vs without
# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] -1

X_test = test.iloc[:, :-2]
y_test = test[['label']] -1 


#%%
# Baseline Model without feature selection
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print(f1_score(y_train, knn_model.predict(X_train) , average='micro')) 
print(f1_score(y_test, knn_model.predict(X_test), average='micro')) 
# 0.965108793614008
# 0.8883617963314357
#%%
# With Feature Selection
train = train.drop(["id"], axis=1)
test = test.drop(["id"], axis=1)

# Split the data into X (features) and y (labels)
X_train = train.drop(["label"], axis=1)

def highCorrFeat(dataframe, threshold):
    """
    Identify highly correlated feature pairs and features to drop based on a given correlation threshold.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing the features.
    threshold (float): The correlation threshold to determine which features are highly correlated. Default is 0.9.
    
    Returns:
    dict: A dictionary of highly correlated feature pairs with their correlation values.
    list: A list of feature columns to drop based on the correlation threshold.
    """
    # Step 1: Calculate the correlation matrix
    correlation_matrix = dataframe.corr().abs()

    # Step 2: Create a mask for the upper triangle
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Step 3: Extract the pairs of highly correlated features
    high_corr_pairs = [(column, row) for column in upper_tri.columns for row in upper_tri.index if upper_tri.loc[row, column] > threshold]

    # Step 4: Store the highly correlated pairs in a dictionary
    res = {}
    for pair in high_corr_pairs:
        corr = correlation_matrix.loc[pair[0], pair[1]]
        res[corr] = [pair[0], pair[1]]

    # Step 5: Find the feature columns that have a correlation greater than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    return res, to_drop
#%%
res_9, to_drop_9 = highCorrFeat(X_train, 0.9)
res_8, to_drop_8 = highCorrFeat(X_train, 0.8)

# Drop the features from both the training and testing set
train_9 = train.drop(columns=to_drop_9)
test_9 = test.drop(columns=to_drop_9)
train_8 = train.drop(columns=to_drop_8)
test_8 = test.drop(columns=to_drop_8)

# Display the results
print("Dropped features:", to_drop_9)
print("Dropped features:", to_drop_8)
print("Original Dataframe shape: ", train.shape)
print("Reduced DataFrame shape of threshold = 0.9:", train_9.shape)
print("Reduced DataFrame shape of threshold = 0.9:", train_8.shape)
# Original Dataframe shape:  (7767, 562)
# Reduced DataFrame shape of threshold = 0.9: (7767, 213)
# Reduced DataFrame shape of threshold = 0.9: (7767, 146)
#%%
X_train = train_8.drop('label', axis=1)
y_train = train_8[['label']]

X_test = test_8.drop('label', axis =1)
y_test = test_8[['label']]
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print(f1_score(y_train, knn_model.predict(X_train) , average='micro')) 
print(f1_score(y_test, knn_model.predict(X_test), average='micro')) 
# 0.9544225569718038
# 0.859898798228969
#%%
X_train = train_9.drop('label', axis=1)
y_train = train_9[['label']]

X_test = test_9.drop('label', axis =1)
y_test = test_9[['label']]
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print(f1_score(y_train, knn_model.predict(X_train) , average='micro')) 
print(f1_score(y_test, knn_model.predict(X_test), average='micro')) 
# 0.956482554396807
# 0.857685009487666
#%%
# With SMOTE (can use cause all data is numerical)
from imblearn.over_sampling import SMOTE
from collections import Counter
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] -1

X_test = test.iloc[:, :-2]
y_test = test[['label']] -1 

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=random_seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(X_train_smote, y_train_smote) 

print(f1_score(y_train_smote, knn_model.predict(X_train_smote) , average='micro')) 
print(f1_score(y_test, knn_model.predict(X_test), average='micro')) 
# 0.9843640196767393
# 0.8915243516761543

#%%

# Find best k, with smote
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

# k-fold to ensure each fold has a similar class distribution
strat_kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)

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
    for train_index, test_index in strat_kfold.split(X_train_smote, y_train_smote):
        # Use iloc to select the correct indices
        X_train_fold, X_test_fold = X_train_smote.iloc[train_index], X_train_smote.iloc[test_index]
        y_train_fold, y_test_fold = y_train_smote.iloc[train_index], y_train_smote.iloc[test_index]
        
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
        train_f1 = f1_score(y_train_fold, y_train_pred, average='micro')
        test_f1 = f1_score(y_test_fold, y_test_pred, average='micro')
        
        fold_train_scores.append(train_f1)
        fold_test_scores.append(test_f1)
    
    # Store the average F1 scores for this k
    train_f1_scores.append(np.mean(fold_train_scores))
    test_f1_scores.append(np.mean(fold_test_scores))

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'Training F1 Score (micro)': train_f1_scores,
    'Test F1 Score (micro)': test_f1_scores
})

# Determine the best k (k with the highest Test Weighted F1)
best_k = k_range[test_f1_scores.index(max(test_f1_scores))]

print(f"Best k: {best_k}")
print(results_df_f1)

# Best k: 1
#      k  Training F1 Score (micro)  Test F1 Score (micro)
# 0    1                   1.000000               0.983896
# 1    2                   0.990721               0.974818
# 2    3                   0.993753               0.981143
# 3    4                   0.988040               0.977161
# 4    5                   0.989674               0.980675
# 5    6                   0.986388               0.977571
# 6    7                   0.987585               0.979679
# 7    8                   0.984917               0.978039
# 8    9                   0.986433               0.979855
# 9   10                   0.984520               0.978625
# 10  11                   0.985314               0.980382
# 11  12                   0.983948               0.978859
# 12  13                   0.984618               0.979269
# 13  14                   0.983466               0.978859
# 14  15                   0.983948               0.978976
# 15  16                   0.982861               0.978391
# 16  17                   0.982991               0.978157
# 17  18                   0.982132               0.977688
# 18  19                   0.981969               0.977747

# Choose k = 5 or 11?
#%%
smote = SMOTE(random_state=random_seed)
strat_kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
k_range = range(1, 20)

# Lists to store cross-validation F1 scores for training and test
train_f1_scores = []
test_f1_scores = []

# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    fold_train_scores = []
    fold_test_scores = []

    # Perform  k-fold split
    for train_index, test_index in strat_kfold.split(X_train, y_train):
        # Split the data into training and validation sets
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        # Apply SMOTE to balance the training fold
        X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)
        
        # Standardize the data within each fold
        scaler = StandardScaler()
        X_train_smote = scaler.fit_transform(X_train_smote)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Train KNN model on the SMOTE-processed training fold
        knn.fit(X_train_smote, y_train_smote)
        
        # Predict on both the training and validation folds
        y_train_pred = knn.predict(X_train_smote)
        y_test_pred = knn.predict(X_test_fold)
        
        # Compute F1 scores for training and test sets
        train_f1 = f1_score(y_train_smote, y_train_pred, average='micro')
        test_f1 = f1_score(y_test_fold, y_test_pred, average='micro')
        
        # Store F1 scores for each fold
        fold_train_scores.append(train_f1)
        fold_test_scores.append(test_f1)

    # Store the average F1 scores for this k
    train_f1_scores.append(np.mean(fold_train_scores))
    test_f1_scores.append(np.mean(fold_test_scores))

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'Training F1 Score (micro)': train_f1_scores,
    'Test F1 Score (micro)': test_f1_scores
})

# Determine the best k (k with the highest Test F1 Score)
best_k = k_range[test_f1_scores.index(max(test_f1_scores))]

print(f"Best k: {best_k}")
print(results_df_f1)

# Once the best k is found, apply the model to the entire training set and final test set

# Apply SMOTE to the entire training set
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardize the entire training and test set
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)

# Train KNN model with the best k found on the whole SMOTE-processed training set
knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn_best.fit(X_train_smote, y_train_smote)

# Evaluate on the final test set
y_test_pred = knn_best.predict(X_test)
test_f1_final = f1_score(y_test, y_test_pred, average='micro')

print(f"Final Test F1 Score with best k ({best_k}): {test_f1_final}")
# Best k: 1
#      k  Training F1 Score (micro)  Test F1 Score (micro)
# 0    1                   1.000000               0.952363
# 1    2                   0.991183               0.933178
# 2    3                   0.993773               0.949271
# 3    4                   0.988561               0.943092
# 4    5                   0.990018               0.949274
# 5    6                   0.987058               0.944768
# 6    7                   0.987865               0.947473
# 7    8                   0.986069               0.943994
# 8    9                   0.986511               0.946828
# 9   10                   0.985145               0.944381
# 10  11                   0.985705               0.946956
# 11  12                   0.984696               0.942836
# 12  13                   0.984781               0.945283
# 13  14                   0.983883               0.942965
# 14  15                   0.983805               0.944124
# 15  16                   0.983193               0.943738
# 16  17                   0.983147               0.943738
# 17  18                   0.982419               0.942836
# 18  19                   0.982191               0.941805

# Final Test F1 Score with best k (1): 0.8244781783681214
#%%

# PCA
from sklearn.decomposition import PCA

# Step 1: Fit PCA to the training data
pca = PCA(n_components=0.9)
pca.fit(X_train_smote)

# Step 2: Transform both training and test data using PCA
X_train_pca = pca.transform(X_train_smote)
X_test_pca = pca.transform(X_test)

# Step 3: Train the KNN model on the transformed data
knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')  # You can adjust k as needed
knn.fit(X_train_pca, y_train_smote)

# Step 4: Predict on both training and test sets
y_train_pred = knn.predict(X_train_pca)
y_test_pred = knn.predict(X_test_pca)

# Step 5: Evaluate the model using classification report and F1 score
print("Training Set Classification Report:")
print(classification_report(y_train_smote, y_train_pred))

print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))

# Optional: Compare F1 weighted scores for training and test sets
train_f1_weighted = f1_score(y_train_smote, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

print(f"Training f1_weighted: {train_f1_weighted:.2f}")
print(f"Test f1_weighted: {test_f1_weighted:.2f}")

# Training Set Classification Report:
#               precision    recall  f1-score   support

#            1       0.99      1.00      0.99      1423
#            2       0.99      0.99      0.99      1423
#            3       1.00      0.99      0.99      1423
#            4       0.87      0.90      0.89      1423
#            5       0.90      0.87      0.89      1423
#            6       1.00      0.99      0.99      1423
#            7       0.99      1.00      1.00      1423
#            8       1.00      1.00      1.00      1423
#            9       1.00      1.00      1.00      1423
#           10       1.00      1.00      1.00      1423
#           11       1.00      1.00      1.00      1423
#           12       1.00      1.00      1.00      1423

#     accuracy                           0.98     17076
#    macro avg       0.98      0.98      0.98     17076
# weighted avg       0.98      0.98      0.98     17076

# Test Set Classification Report:
#               precision    recall  f1-score   support

#            1       0.85      0.96      0.90       496
#            2       0.87      0.88      0.88       471
# ...
# weighted avg       0.87      0.86      0.86      3162

# Training f1_weighted: 0.98
# Test f1_weighted: 0.86

