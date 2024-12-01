# %%
import csv
import math
import random
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler


random_seed = 31

# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] - 1

X_test = test.iloc[:, :-2]
y_test = test[['label']] - 1 
#%%
# Baseline model (default parameters, no fs, no oversampling)
knn = neighbors.KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p=2) # default k = 5, dist = minkowski
knn_model = knn.fit(X_train, y_train) 

print(f1_score(y_train, knn_model.predict(X_train) , average='weighted')) # 0.975782440942216
print(f1_score(y_test, knn_model.predict(X_test), average='weighted'))  # 0.8827831477295671
print(f1_score(y_train, knn_model.predict(X_train) , average='macro')) # 0.9109836775301784
print(f1_score(y_test, knn_model.predict(X_test), average='macro'))  # 0.8135337752124174

print(classification_report(y_test, knn_model.predict(X_test)))
#               precision    recall  f1-score   support

#            0       0.85      0.98      0.91       496
#            1       0.87      0.90      0.88       471
#            2       0.95      0.78      0.85       420
#            3       0.90      0.78      0.84       508
#            4       0.83      0.92      0.87       556
#            5       1.00      0.99      1.00       545
#            6       0.89      0.74      0.81        23
#            7       1.00      1.00      1.00        10
#            8       0.61      0.88      0.72        32
#            9       0.65      0.88      0.75        25
#           10       0.79      0.47      0.59        49
#           11       0.85      0.41      0.55        27

#     accuracy                           0.88      3162
#    macro avg       0.85      0.81      0.81      3162
# weighted avg       0.89      0.88      0.88      3162

#%%
# Baseline Model (default params, with feature selection, without oversampling)

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
X_train_8 = train_8.drop('label', axis=1)
y_train_8 = train_8[['label']]

X_test_8 = test_8.drop('label', axis =1)
y_test_8 = test_8[['label']]
knn = neighbors.KNeighborsClassifier(n_neighbors = 5, metric='minkowski')
knn_model = knn.fit(X_train_8, y_train_8) 

print(f1_score(y_train_8, knn_model.predict(X_train_8) , average='weighted')) # 0.9666702857272359
print(f1_score(y_test_8, knn_model.predict(X_test_8), average='weighted'))  # 0.8484672390717394
print(f1_score(y_train_8, knn_model.predict(X_train_8) , average='macro')) # 0.9024469531032707
print(f1_score(y_test_8, knn_model.predict(X_test_8), average='macro'))  # 0.771609290370255


#%%
X_train_9 = train_9.drop('label', axis=1)
y_train_9 = train_9[['label']]

X_test_9 = test_9.drop('label', axis =1)
y_test_9 = test_9[['label']]
knn = neighbors.KNeighborsClassifier(n_neighbors = 5, metric='minkowski')
knn_model = knn.fit(X_train_9, y_train_9) 

print(f1_score(y_train_9, knn_model.predict(X_train_9) , average='micro'))  # 0.9720612849233938
print(f1_score(y_test_9, knn_model.predict(X_test_9), average='micro'))  # 0.8633776091081594
#%%

# Baseline Model (default params, without feature selection, with oversampling)

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
knn = neighbors.KNeighborsClassifier(n_neighbors = 5, metric='minkowski')
knn_model = knn.fit(X_train_smote, y_train_smote) 

print(f1_score(y_train_smote, knn_model.predict(X_train_smote) , average='weighted'))  # 0.9906625022317752
print(f1_score(y_test, knn_model.predict(X_test), average='weighted'))  # 0.8875149031340788
print(f1_score(y_train_smote, knn_model.predict(X_train_smote) , average='macro'))  # 0.9906625022317751
print(f1_score(y_test, knn_model.predict(X_test), average='macro'))  # 0.8013433442830328

#%%

# Baseline Model (default params, with feature selection (0.8), with oversampling)

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=random_seed)
X_train_smote_8, y_train_smote_8 = smote.fit_resample(X_train_8, y_train_8)

knn = neighbors.KNeighborsClassifier(n_neighbors = 5, metric='minkowski')
knn_model = knn.fit(X_train_smote_8, y_train_smote_8) 

print(f1_score(y_train_smote_8, knn_model.predict(X_train_smote_8) , average='weighted'))  # 0.9851294673969593
print(f1_score(y_test_8, knn_model.predict(X_test_8), average='weighted'))  # 0.8475332153606772
print(f1_score(y_train_smote_8, knn_model.predict(X_train_smote_8) , average='macro'))  # 0.9851294673969594
print(f1_score(y_test_8, knn_model.predict(X_test_8), average='macro'))  # 0.7672577042126535


#%%

# Tuning oversampled model (find best k, euclidean)
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
    for train_index, test_index in kfold.split(X_train_smote, y_train_smote):
        X_train_fold, X_test_fold = X_train_smote.iloc[train_index], X_train_smote.iloc[test_index]
        y_train_fold, y_test_fold = y_train_smote.iloc[train_index], y_train_smote.iloc[test_index]
        
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
# 0    1           1.000000       0.983896
# 1    2           0.990721       0.974818
# 2    3           0.993753       0.981143
# 3    4           0.988040       0.977161
# 4    5           0.989674       0.980675
# 5    6           0.986388       0.977571
# 6    7           0.987585       0.979679
# 7    8           0.984917       0.978039
# 8    9           0.986433       0.979855
# 9   10           0.984520       0.978625
# 10  11           0.985314       0.980382
# 11  12           0.983948       0.978859
# 12  13           0.984618       0.979269
# 13  14           0.983466       0.978859
# 14  15           0.983948       0.978976
# 15  16           0.982861       0.978391
# 16  17           0.982991       0.978157
# 17  18           0.982132       0.977688
# 18  19           0.981969       0.977747
# 19  20           0.981345       0.977922

# Choose k = 11 to balance training and test score (for euclidean)
#%%

# Tuning oversampled model (find best k, minkowski)
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
    for train_index, test_index in kfold.split(X_train_smote, y_train_smote):
        X_train_fold, X_test_fold = X_train_smote.iloc[train_index], X_train_smote.iloc[test_index]
        y_train_fold, y_test_fold = y_train_smote.iloc[train_index], y_train_smote.iloc[test_index]
        
        # Standardize the data for each fold
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Initialize and fit KNN model
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
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
# 0    1           1.000000       0.983896
# 1    2           0.990721       0.974818
# 2    3           0.993753       0.981143
# 3    4           0.988040       0.977161
# 4    5           0.989674       0.980675
# 5    6           0.986388       0.977571
# 6    7           0.987585       0.979679
# 7    8           0.984917       0.978039
# 8    9           0.986433       0.979855
# 9   10           0.984520       0.978625
# 10  11           0.985314       0.980382
# 11  12           0.983948       0.978859
# 12  13           0.984618       0.979269
# 13  14           0.983466       0.978859
# 14  15           0.983948       0.978976
# 15  16           0.982861       0.978391
# 16  17           0.982991       0.978157
# 17  18           0.982132       0.977688
# 18  19           0.981969       0.977747
# 19  20           0.981345       0.977922


# Choose k = 11 to balance training and test score (same as euclidean)

#%%
# Tuning oversampled model (find best k, manhattan)
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
    for train_index, test_index in kfold.split(X_train_smote, y_train_smote):
        X_train_fold, X_test_fold = X_train_smote.iloc[train_index], X_train_smote.iloc[test_index]
        y_train_fold, y_test_fold = y_train_smote.iloc[train_index], y_train_smote.iloc[test_index]
        
        # Standardize the data for each fold
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Initialize and fit KNN model
        knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
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
# 0    1           1.000000       0.993207
# 1    2           0.995784       0.986296
# 2    3           0.997313       0.990337
# 3    4           0.994079       0.987116
# 4    5           0.994645       0.989342
# 5    6           0.992738       0.988288
# 6    7           0.992621       0.988171
# 7    8           0.991463       0.987527
# 8    9           0.991509       0.987819
# 9   10           0.990565       0.987000
# 10  11           0.990396       0.986941
# 11  12           0.989661       0.986121
# 12  13           0.989999       0.987000
# 13  14           0.989511       0.986297
# 14  15           0.989680       0.986999
# 15  16           0.988925       0.986414
# 16  17           0.989212       0.986531
# 17  18           0.988600       0.986121
# 18  19           0.988769       0.985886
# 19  20           0.988450       0.986062
#%%

# Therefore, best parameters: manhattan, k = 10(?)

#%%
# Identify misclassified instances (from tuned oversampled model)
knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
knn.fit(X_train_smote, y_train_smote.squeeze())  # .squeeze() to pass as Series

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


# print("Misclassified Instances:\n", misclassified_df)
print("Total Misclassified Instances:", len(misclassified_df))
# misclassified_df.to_csv('knn_misclassifications.csv', index=False)

#      Index  True Label  Predicted Label
# 0       17           3                4
# 1       20           3                4
# 2       28           3                4
# 3       49          10                8
# 4       50          10                8
# ..     ...         ...              ...
# 307   3007           4                3
# 308   3010           4                3
# 309   3113           2                0
# 310   3130           2                0
# 311   3161           1                0

# [312 rows x 3 columns]


# Total Misclassified Instances: 312


