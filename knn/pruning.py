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
# Make the classes balanced (all classes have same number of records/occurences)

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
X_res = balanced_combined.iloc[:, :-2]  # All columns except the label
y_res = balanced_combined[['label']]    # The label column

# Check the new distribution of labels
print(y_res['label'].value_counts().sort_index())
#%%
from sklearn.model_selection import train_test_split
# training: 70%
# test: 30%

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print('After pruning for majority class to 1:1')
print('kNN accuracy for training set: %f' % knn_model.score(X_train, y_train))
print('kNN accuracy for test set: %f' % knn_model.score(X_test, y_test))

# After pruning for majority class to 1:1
# kNN accuracy for training set: 0.819495
# kNN accuracy for test set: 0.756303
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

# Best k: 1
#%%
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

# Best k: 1 (without random seed)
# Best k: 11 (with random seed)
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

# SVM
# combine original test and train then split again on 70:30

combined_X = combined.iloc[:, :-2] # remove id and label col
combined_y = combined[['label']]

X_train, X_test, y_train, y_test = train_test_split(combined_X, combined_y, test_size=0.3, random_state=42)


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, svc.predict(X_test))) # 0.9594388533089356
print(accuracy_score(y_train, svc.predict(X_train))) # 0.9669281045751634
#%%

# SVM
# using their train and test


ori_X_train = train.iloc[:, :-2]
ori_y_train = train[['label']]

ori_X_test = test.iloc[:, :-2]
ori_y_test = test[['label']]

svc = SVC()
svc.fit(ori_X_train, ori_y_train)
print(accuracy_score(y_test, svc.predict(X_test))) # 0.9640134187252211
print(accuracy_score(y_train, svc.predict(X_train))) # 0.9605228758169935
#%%

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.03)
sel.fit(train)

mask = sel.get_support()
# print(mask)


reduced_train = train.iloc[:, :-2].loc[: , mask]
reduced_test = test.iloc[:, :-2].loc[:, mask]


# reduced_X_train = reduced_train.iloc[:, :-2]
# reduced_y_train = reduced_test.loc[:, mask]

# reduced_X_test = reduced_test.iloc[:, :-2]
# reduced_y_test = reduced_test.loc[:, mask]
#%%
from sklearn.feature_selection import VarianceThreshold

# Set the threshold for variance
threshold = 0.03

# Initialize the VarianceThreshold object with the threshold
selector = VarianceThreshold(threshold=threshold)

# Apply the selector to your dataset (it returns only the features that meet the threshold)
X_high_variance = selector.fit_transform(ori_X_train)


# To get the column names of the selected features, use the selector's support attribute
selected_columns = ori_X_train.columns[selector.get_support()]


# Create a new DataFrame with only the high variance features
reduced_X = pd.DataFrame(X_high_variance, columns=selected_columns)


# Check the shape of the new DataFrame to confirm the reduction in features
# print(reduced_X.shape)


reduced_X_train = reduced_X
reduced_y_train = ori_y_train

reduced_X_test = pd.DataFrame(ori_X_test, columns=selected_columns)
reduced_y_test = ori_y_test

knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(reduced_X_train, reduced_y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(reduced_X_train, reduced_y_train))
print('kNN accuracy for test set: %f' % knn_model.score(reduced_X_test, reduced_y_test))
# kNN accuracy for training set: 0.966268
# kNN accuracy for test set: 0.886781

#%%

# Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming you already have X_train_new and y_train_new
# Split the dataset into training and testing sets
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train_bag = train.iloc[:, :-2] # remove id and label col
y_train_bag = train[['label']] 

X_test_bag = test.iloc[:, :-2]
y_test_bag = test[['label']]

# Initialize kNN classifier (the base learner)
knn = KNeighborsClassifier(n_neighbors=10)

# Initialize BaggingClassifier using kNN as the base estimator
bagging_model = BaggingClassifier(estimator=knn, n_estimators=50, random_state=42)

# Fit the bagging model to the training data
bagging_model.fit(X_train_bag, y_train_bag)

# Predict on the test set
y_pred_bag = bagging_model.predict(X_test_bag)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test_bag, y_pred_bag)
print(f'Bagging kNN Accuracy: {accuracy:.4f}')

# Print classification report
print(classification_report(y_test_bag, y_pred_bag))

# Bagging kNN Accuracy: 0.8918
#               precision    recall  f1-score   support

#            1       0.85      0.98      0.91       496
#            2       0.87      0.92      0.90       471
#            3       0.96      0.78      0.86       420
#            4       0.91      0.80      0.85       508
#            5       0.84      0.93      0.88       556
#            6       1.00      0.99      1.00       545
#            7       0.89      0.74      0.81        23
#            8       1.00      0.90      0.95        10
#            9       0.62      0.91      0.73        32
#           10       0.61      0.80      0.69        25
#           11       0.82      0.47      0.60        49
#           12       0.71      0.37      0.49        27

#     accuracy                           0.89      3162
#    macro avg       0.84      0.80      0.81      3162
# weighted avg       0.90      0.89      0.89      3162

#%%
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Try different values for n_neighbors (e.g., 3, 7, 10)

# 1. K-Fold Cross-Validation for Training Set
# Perform 5-fold cross-validation
random_seed = 31
# Stratified k-fold to ensure each fold has a similar class distribution
strat_kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state=random_seed)

cv_scores = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='f1_weighted')

# Print the cross-validation results
print("Cross-Validation f1_weighted Scores:", cv_scores)
print("Mean CV f1_weighted: {:.2f}".format(cv_scores.mean()))

# 2. Train the KNN Model on the Training Data and Evaluate on Test Data
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Evaluate the model using classification report for both training and test sets
print("Training Set Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))

# Compare f1_weighted
train_f1_weighted = f1_score(y_train, knn.predict(X_train), average='weighted')
test_f1_weighted = f1_score(y_test, knn.predict(X_test), average='weighted')

print(f"Training f1_weighted: {train_f1_weighted:.2f}")
print(f"Test f1_weighted: {test_f1_weighted:.2f}")

# 3. Adjust Model Complexity (Regularization by changing k value)
# You can loop over different k values to see which works best.
k_values = range(1, 20)
for k in k_values:
    knn = neighbors.KNeighborsClassifier(n_neighbors = k, metric='euclidean')
    knn.fit(X_train, y_train)
    
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"k={k} | Train f1_weighted: {train_f1_weighted:.2f} | Test f1_weighted: {test_f1_weighted:.2f}")

# k=10 | Train f1_weighted: 0.96 | Test f1_weighted: 0.89
# multiple k values with same train and test f1 score as k = 10
#%%
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=10), X_train, y_train, cv=strat_kfold, scoring='f1_weighted')
print("Cross-Validation f1_weighted: ", cv_scores.mean())

# since this cv f1_weighted is close to the test f1_weighted means little overfitting

# Cross-Validation f1_weighted:  0.9466366976309812