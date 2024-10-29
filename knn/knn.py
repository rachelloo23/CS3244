# %%
import csv
import math
import random
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler


random_seed = 31

# First part: compare with feature selection and without
# Second part: oversample vs without
# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] - 1

X_test = test.iloc[:, :-2]
y_test = test[['label']] - 1 
#%%
# Tuning baseline model (find best k)
kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)

# Range of k values to evaluate
k_range = range(1, 21)  # adjust the range if necessary

# Lists to store average F1 scores for each k value
train_f1_scores = []
test_f1_scores = []

# Loop over each k
for k in k_range:
    # Store F1 scores for each fold
    fold_train_scores = []
    fold_test_scores = []
    
    # Perform K-Fold cross-validation
    for train_index, test_index in kfold.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        # Standardize the data for each fold
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Initialize and fit KNN model
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train_fold, y_train_fold)
        
        # Predict and calculate F1 score
        y_train_pred = knn.predict(X_train_fold)
        y_test_pred = knn.predict(X_test_fold)
        
        # Compute F1 scores for training and test sets
        fold_train_scores.append(f1_score(y_train_fold, y_train_pred, average='micro'))
        fold_test_scores.append(f1_score(y_test_fold, y_test_pred, average='micro'))
    
    # Store the average F1 score across folds for each k
    train_f1_scores.append(np.mean(fold_train_scores))
    test_f1_scores.append(np.mean(fold_test_scores))

# Find the best k based on the highest test F1 score
best_k = k_range[np.argmax(test_f1_scores)]

# Display results in a DataFrame
results_df = pd.DataFrame({
    'k': list(k_range),
    'Training F1 Score': train_f1_scores,
    'Test F1 Score': test_f1_scores
})

print("Best k:", best_k)
print("Results DataFrame:\n", results_df)
# Best k: 1
# Results DataFrame:
#       k  Training F1 Score  Test F1 Score
# 0    1           1.000000       0.956868
# 1    2           0.975709       0.936397
# 2    3           0.984407       0.951849
# 3    4           0.973506       0.946954
# 4    5           0.971761       0.948886
# 5    6           0.968428       0.945667
# 6    7           0.967283       0.946696
# 7    8           0.964436       0.945280
# 8    9           0.962176       0.945280
# 9   10           0.960345       0.943864
# 10  11           0.958600       0.944379
# 11  12           0.957927       0.944637
# 12  13           0.955124       0.940259
# 13  14           0.955281       0.941675
# 14  15           0.951876       0.940517
# 15  16           0.952677       0.940645
# 16  17           0.949258       0.938200
# 17  18           0.949816       0.938586
# 18  19           0.947127       0.937942
# 19  20           0.948257       0.938585

#%%
# (tuned) Baseline Model (without feature selection, without oversampling)
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print(f1_score(y_train, knn_model.predict(X_train) , average='micro')) 
print(f1_score(y_test, knn_model.predict(X_test), average='micro')) 
# 1.0
# 0.8646426312460468

print(classification_report(y_test, knn_model.predict(X_test)))
#        precision    recall  f1-score   support

#            0       0.85      0.96      0.90       496
#            1       0.88      0.90      0.89       471
#            2       0.92      0.77      0.84       420
#            3       0.80      0.76      0.78       508
#            4       0.80      0.84      0.82       556
#            5       1.00      0.99      0.99       545
#            6       1.00      0.74      0.85        23
#            7       0.71      1.00      0.83        10
#            8       0.64      0.84      0.73        32
#            9       0.62      0.80      0.70        25
#           10       0.80      0.65      0.72        49
#           11       0.75      0.44      0.56        27

#     accuracy                           0.86      3162
#    macro avg       0.81      0.81      0.80      3162
# weighted avg       0.87      0.86      0.86      3162

#%%
# Identify misclassified instances (from tuned baseline model)
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train.squeeze())  # .squeeze() to pass as Series

# Make predictions
y_pred = knn.predict(X_test)

# Identify misclassified instances and store their indices
y_test_series = y_test.squeeze()  # Convert y_test to Series for comparison
misclassified_indices = y_test_series[y_test_series != y_pred].index

# Create DataFrame with misclassifications
misclassified_df = pd.DataFrame({
    'Index': misclassified_indices,
    'True Label': y_test_series[misclassified_indices].values,
    'Predicted Label': y_pred[misclassified_indices]
})

# Display results
print("Misclassified Instances:\n", misclassified_df)
print("Total Misclassified Instances:", len(misclassified_df))
# Misclassified Instances:
#       Index  True Label  Predicted Label
# 0        3           4                3
# 1        8           4                3
# 2       11           4                3
# 3       12           4                3
# 4       16           6                0
# ..     ...         ...              ...
# 423   3111           2                1
# 424   3113           2                0
# 425   3132           2                0
# 426   3144           2                0
# 427   3161           1                0

# [428 rows x 3 columns]
# Total Misclassified Instances: 428

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
