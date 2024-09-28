# %%
import csv
import math
import random
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]
#%%
# Concatenate the labels from train and test datasets
combined = pd.concat([train, test])

# Get the label distribution in table form
label_distribution = combined['label'].value_counts().sort_index()

# Print the table
print(label_distribution)

# Plot the histogram of the label distribution
plt.figure(figsize=(8, 6))
combined['label'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Labels in Combined Train and Test Sets')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()

# Label 8: 33 (least)
#%%
# Prune so that all labels have only 33 occurences

# Set the target number of occurrences per label
target_samples = 33

# Function to prune or oversample data for each label
def prune_except_label_8(df, label_col, target_count):
    balanced_df = pd.DataFrame()  # Empty dataframe to store results
    for label in df[label_col].unique():
        label_df = df[df[label_col] == label]
        if len(label_df) > target_count and label != 8:
            # Randomly sample to get exactly target_count samples (for labels with more than target_count except label 8)
            label_df = label_df.sample(target_count)
        # For label 8, keep all original samples (no pruning)
        balanced_df = pd.concat([balanced_df, label_df])
    return balanced_df

# Prune except for label 8
balanced_combined = prune_except_label_8(combined, 'label', target_samples)

# Separate the features and labels after balancing
X_res = balanced_combined.iloc[:, :-1]  # All columns except the label
y_res = balanced_combined[['label']]    # The label column

# Check the new distribution of labels
print(y_res['label'].value_counts().sort_index())
#%%
from sklearn.model_selection import train_test_split
# training: 70%
# test: 30%

X = combined.iloc[:, :-2] # remove id and label col
y = combined[['label']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = neighbors.KNeighborsClassifier(n_neighbors = 1, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(X_train, y_train))
print('kNN accuracy for test set: %f' % knn_model.score(X_test, y_test))
#%%

# Choosing best k (based on 10-fold CV, f1_weighted)

cv_scores = []

k_range = range(1, 20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # Perform 10-fold cross-validation and take the weighted f1
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='f1_weighted')
    # For multiclass:
    # 'f1_macro': Averages the F1 score for each class without considering class imbalance.
    # 'f1_weighted': Averages the F1 score for each class, weighted by the number of samples in each class.
    cv_scores.append(scores.mean())

# Determine the best k (k with the highest Weighted f1)
best_k = k_range[cv_scores.index(max(cv_scores))]

print(f"Best k: {best_k}")

# Plot k values vs cross-validation scores
plt.plot(k_range, cv_scores)
plt.xlabel('k')
plt.ylabel('Cross-Validated Weighted f1')
plt.title('K value vs Weighted f1')
plt.show()

# Table form
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'F1 Score': cv_scores
})

print(results_df_f1)
#%%
# Choosing best k (based on Stratified 10-fold CV, f1_weighted)
# random_seed = 31
# Stratified k-fold to ensure each fold has a similar class distribution
# strat_kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state=random_seed)
strat_kfold = StratifiedKFold(n_splits=10)
# List to store cross-validation F1 scores
cv_f1_scores = []
temp_1 = []
k_range = range(1, 20)

# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,  metric='euclidean')
    f1_scores = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='f1_weighted')
    temp_1.append(f1_scores)
    cv_f1_scores.append(f1_scores.mean())

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'F1 Score (weighted)': cv_f1_scores
})
# Determine the best k (k with the highest Weighted f1)
best_k = k_range[cv_f1_scores.index(max(cv_f1_scores))]

print(f"Best k: {best_k}")
print(results_df_f1)
print(temp_1)
#%%

# Drop Related features (corr > 0.8)

import pandas as pd

# Load your dataset
combined_X = combined.iloc[:, :-2] # remove id and label col
combined_y = combined[['label']]
# Calculate the correlation matrix
correlation_matrix = combined_X.corr().abs()  # Absolute values to consider both positive and negative correlations

# Define a threshold for high correlation
threshold = 0.8

# Find features that are correlated above the threshold
to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        # Check for high correlation
        if correlation_matrix.iloc[i, j] > threshold:
            colname = correlation_matrix.columns[i]
            to_drop.add(colname)

# Remove the correlated features from the original DataFrame
combined_X_reduced = combined_X.drop(columns=to_drop)

# Display the results
print("Dropped features:", to_drop)
print("Original Dataframe shape: ", combined_X.shape)
print("Reduced DataFrame shape:", combined_X_reduced.shape)
#%%
# Use reduced features to test knn
X_train, X_test, y_train, y_test = train_test_split(combined_X_reduced, combined_y, test_size=0.3, random_state=42)

knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(X_train, y_train))
print('kNN accuracy for test set: %f' % knn_model.score(X_test, y_test))
#%%

# do cv on this reduced dataset
# Choosing best k (based on Stratified 10-fold CV, f1_weighted)
random_seed = 31
# Stratified k-fold to ensure each fold has a similar class distribution
strat_kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state=random_seed)
# strat_kfold = StratifiedKFold(n_splits=10)
# List to store cross-validation F1 scores
cv_f1_scores = []
temp_1 = []
k_range = range(1, 20)

# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,  metric='euclidean')
    f1_scores = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='f1_weighted')
    temp_1.append(f1_scores)
    cv_f1_scores.append(f1_scores.mean())

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'F1 Score (weighted)': cv_f1_scores
})
# Determine the best k (k with the highest Weighted f1)
best_k = k_range[cv_f1_scores.index(max(cv_f1_scores))]

print(f"Best k: {best_k}")
print(results_df_f1)
print(temp_1)
# Best k: 7
#      k  F1 Score (weighted)
# 0    1             0.924653
# 1    2             0.891677
# 2    3             0.925440
# 3    4             0.913300
# 4    5             0.924575
# 5    6             0.919073
# 6    7             0.926966
# 7    8             0.917937
# 8    9             0.923414
# 9   10             0.918825
# 10  11             0.920742
# 11  12             0.916255
# 12  13             0.917646
# 13  14             0.913203
# 14  15             0.916930
# 15  16             0.910478
# 16  17             0.913287
# 17  18             0.911897
# 18  19             0.911481
#%%
