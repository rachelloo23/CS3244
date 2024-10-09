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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]
#%%

# LDA (check if data is linearly separable)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Train LDA
lda = LDA()
lda.fit(X_train, y_train)

# Check accuracy on training data
y_train_pred = lda.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)

print(f"Training accuracy with LDA: {accuracy:.2f}")
# Training accuracy with LDA: 0.98 -> data is nearly linearly separable
#%%
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

y_train_pred_rbf = svm_rbf.predict(X_train)
accuracy_rbf = accuracy_score(y_train, y_train_pred_rbf)

print(f"Training accuracy with RBF Kernel: {accuracy_rbf:.2f}")
# Training accuracy with RBF Kernel: 0.97

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
# training: 70%
# test: 30%

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, stratify=y_res,test_size=0.3, random_state=42)

knn = neighbors.KNeighborsClassifier(n_neighbors = 3, metric='euclidean')
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

k_range = range(1, 12)

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

# Original Dataframe shape:  (10929, 561)
# Reduced DataFrame shape: (10929, 148)
#%%
# Use reduced features to test knn
X_train, X_test, y_train, y_test = train_test_split(combined_X_reduced, combined_y, test_size=0.3, random_state=42)

knn = neighbors.KNeighborsClassifier(n_neighbors = 7, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print(f1_score(y_train, knn_model.predict(X_train) , average='weighted')) # 0.9506891426737768
print(f1_score(y_test, knn_model.predict(X_test), average='weighted')) # 0.9269573991573669
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
from sklearn.svm import SVC

# using original train and test (given from dataset)

ori_X_train = train.iloc[:, :-2]
ori_y_train = train[['label']]

ori_X_test = test.iloc[:, :-2]
ori_y_test = test[['label']]

svc = SVC()
svc.fit(ori_X_train, ori_y_train)
# print(accuracy_score(ori_y_test, svc.predict(ori_X_test))) # 0.9367488931056294
# print(accuracy_score(ori_y_train, svc.predict(ori_X_train))) # 0.9716750354062057

# print(f1_score(ori_y_test, svc.predict(ori_X_test), average='weighted')) # 0.9360546876940178
# print(f1_score(ori_y_train, svc.predict(ori_X_train), average='weighted')) # 0.971551119209504

#%%

# Using reduced dataset (removed high correlation features > 0.8)

X_train, X_test, y_train, y_test = train_test_split(combined_X_reduced, combined_y, test_size=0.3, random_state=42)

svc = SVC()
svc.fit(X_train, y_train)

print(f1_score(y_test, svc.predict(X_test), average='weighted')) # 0.959771871547412
print(f1_score(y_train, svc.predict(X_train), average='weighted')) # 0.9749170994905294

#%%

# Find best variance threshold value
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np

# Range of variance thresholds to try
thresholds = np.arange(0.01, 0.2, 0.01)

# Stratified K-Fold for cross-validation
strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=31)

# Variable to store the best score and corresponding threshold
best_f1_score = 0
best_threshold = 0
cv_f1_scores = []

# Loop over different variance thresholds
for threshold in thresholds:
    # Perform feature selection with the given threshold
    selector = VarianceThreshold(threshold=threshold)
    X_high_variance = selector.fit_transform(ori_X_train)
    
    # Train a KNN model using cross-validation
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    f1_scores = cross_val_score(knn, X_high_variance, y_train, cv=strat_kfold, scoring='f1_weighted')
    
    # Calculate the mean F1 weighted score
    mean_f1_score = f1_scores.mean()
    cv_f1_scores.append(mean_f1_score)
    
    # Keep track of the best F1 score and threshold
    if mean_f1_score > best_f1_score:
        best_f1_score = mean_f1_score
        best_threshold = threshold

# Print the best threshold and corresponding F1 score
print(f"Best variance threshold: {best_threshold}")
print(f"Best cross-validated F1 weighted score: {best_f1_score:.4f}")

# Optional: Plot F1 scores across different thresholds
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(thresholds, cv_f1_scores, marker='o', color='b')
plt.title('F1 Weighted Score vs Variance Threshold')
plt.xlabel('Variance Threshold')
plt.ylabel('F1 Weighted Score')
plt.grid(True)
plt.show()

# Best variance threshold: 0.04
# Best cross-validated F1 weighted score: 0.9523

#%%

from sklearn.feature_selection import VarianceThreshold

# Set the threshold for variance
threshold = 0.04

# Initialize the VarianceThreshold object with the threshold
selector = VarianceThreshold(threshold=threshold)

# Apply the selector to your dataset (it returns only the features that meet the threshold)
X_high_variance = selector.fit_transform(ori_X_train)


# To get the column names of the selected features, use the selector's support attribute
selected_columns = ori_X_train.columns[selector.get_support()]


# Create a new DataFrame with only the high variance features
reduced_X = pd.DataFrame(X_high_variance, columns=selected_columns)


# Check the shape of the new DataFrame to confirm the reduction in features
print(reduced_X.shape) 
# (7767, 386)

reduced_X_train = reduced_X
reduced_y_train = ori_y_train

reduced_X_test = pd.DataFrame(ori_X_test, columns=selected_columns)
reduced_y_test = ori_y_test

knn = neighbors.KNeighborsClassifier(n_neighbors = 7, metric='euclidean')
knn_model = knn.fit(reduced_X_train, reduced_y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(reduced_X_train, reduced_y_train))
print('kNN accuracy for test set: %f' % knn_model.score(reduced_X_test, reduced_y_test))
# kNN accuracy for training set: 0.970774
# kNN accuracy for test set: 0.890259

print(f1_score(reduced_y_train, knn_model.predict(reduced_X_train) , average='weighted')) # 0.9704876645013684
print(f1_score(reduced_y_test, knn_model.predict(reduced_X_test), average='weighted')) # 0.8885734965767056


#%%

# Bagging (after removing features with var < 0.04)
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming you already have X_train_new and y_train_new
# Split the dataset into training and testing sets

X_train_bag = reduced_X_train
y_train_bag = reduced_y_train

X_test_bag = reduced_X_test
y_test_bag = reduced_y_test

# Initialize kNN classifier (the base learner)
knn = KNeighborsClassifier(n_neighbors=7)

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

# Without removing low variance features
# Bagging kNN Accuracy: 0.8925
#               precision    recall  f1-score   support

#            1       0.86      0.98      0.91       496
#            2       0.88      0.92      0.90       471
#            3       0.96      0.79      0.86       420
#            4       0.90      0.80      0.85       508
#            5       0.84      0.93      0.88       556
#            6       1.00      0.99      1.00       545
#            7       0.89      0.74      0.81        23
#            8       1.00      1.00      1.00        10
#            9       0.62      0.91      0.73        32
#           10       0.64      0.84      0.72        25
#           11       0.80      0.49      0.61        49
#           12       0.79      0.41      0.54        27

#     accuracy                           0.89      3162
#    macro avg       0.85      0.82      0.82      3162
# weighted avg       0.90      0.89      0.89      3162


# After removing low variance features
# Bagging kNN Accuracy: 0.8922
#               precision    recall  f1-score   support

#            1       0.86      0.98      0.91       496
#            2       0.88      0.91      0.90       471
#            3       0.94      0.79      0.86       420
#            4       0.90      0.79      0.84       508
#            5       0.83      0.92      0.87       556
#            6       1.00      0.99      1.00       545
#            7       0.94      0.74      0.83        23
#            8       1.00      1.00      1.00        10
#            9       0.67      0.91      0.77        32
#           10       0.67      0.88      0.76        25
#           11       0.85      0.59      0.70        49
#           12       0.80      0.44      0.57        27

#     accuracy                           0.89      3162
#    macro avg       0.86      0.83      0.83      3162
# weighted avg       0.90      0.89      0.89      3162
#%%

# Find best k after removing highly correlated features

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report

# Set random seed for reproducibility
random_seed = 31

strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

# List to store cross-validation F1 scores for different k values
cv_f1_scores = []
train_f1_scores = []
test_f1_scores = []
k_range = range(1, 20)
X_train, X_test, y_train, y_test = train_test_split(combined_X_reduced, combined_y, test_size=0.3, random_state=42)


# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    
    
    # Perform cross-validation and get the mean f1_weighted score
    f1_scores = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='f1_weighted')
    cv_f1_scores.append(f1_scores.mean())
    
    # Train the KNN model on the entire training set and evaluate on both train and test sets
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    # Store f1_weighted score for training and test sets
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    train_f1_scores.append(train_f1)
    test_f1_scores.append(test_f1)
    
    # Print k, train, and test F1 scores
    print(f"k={k} | Train F1 (weighted): {train_f1:.2f} | Test F1 (weighted): {test_f1:.2f}")

# Create a DataFrame to store k values and corresponding F1 scores for train, test, and cross-validation
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'Cross-Validated F1 (weighted)': cv_f1_scores,
    'Train F1 (weighted)': train_f1_scores,
    'Test F1 (weighted)': test_f1_scores
})

# Determine the best k based on cross-validation score
best_k = k_range[cv_f1_scores.index(max(cv_f1_scores))]
print(f"Best k found through cross-validation: {best_k}")

# Plot train, test, and cross-validated F1 scores to visually inspect overfitting
import matplotlib.pyplot as plt

plt.plot(k_range, cv_f1_scores, label='Cross-Validated F1', marker='o')
plt.plot(k_range, train_f1_scores, label='Train F1', marker='o')
plt.plot(k_range, test_f1_scores, label='Test F1', marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('F1 Score (weighted)')
plt.legend()
plt.title('F1 Scores for Train, Test, and Cross-Validation')
plt.show()

# Best k found through cross-validation: 7
# k=7 | Train F1 (weighted): 0.95 | Test F1 (weighted): 0.93


#%%

# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=160)

pca.fit(X_train)

explained_var = pca.explained_variance_ratio_.cumsum()

# sum(explained_var < 0.99) 
# 160

#%%
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

# Step 1: Fit PCA to the training data
pca = PCA(n_components=160)
pca.fit(X_train)

# Step 2: Transform both training and test data using PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 3: Train the KNN model on the transformed data
knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')  # You can adjust k as needed
knn.fit(X_train_pca, y_train)

# Step 4: Predict on both training and test sets
y_train_pred = knn.predict(X_train_pca)
y_test_pred = knn.predict(X_test_pca)

# Step 5: Evaluate the model using classification report and F1 score
print("Training Set Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))

# Optional: Compare F1 weighted scores for training and test sets
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

print(f"Training f1_weighted: {train_f1_weighted:.2f}")
print(f"Test f1_weighted: {test_f1_weighted:.2f}")

# k: 10
# Training f1_weighted: 0.96
# Test f1_weighted: 0.88

# k: 7
# Training f1_weighted: 0.97
# Test f1_weighted: 0.89

#%%

# Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')  # You can adjust k as needed
knn.fit(X_train, y_train) 


confusion_matrix(y_true=y_test, y_pred=knn.predict(X_test))
#%% 
# ROC
