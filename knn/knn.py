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

# %%
# Get the machine learning algorithm k-NN (using k = 1)

knn = neighbors.KNeighborsClassifier(n_neighbors = 1, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(X_train, y_train))
print('kNN accuracy for test set: %f' % knn_model.score(X_test, y_test))
#%%
# Choosing best k (based on 10-fold CV, accuracy)

cv_scores = []

k_range = range(1, 20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Perform 10-fold cross-validation and take the mean accuracy
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Determine the best k (k with the highest accuracy)
best_k = k_range[cv_scores.index(max(cv_scores))]

print(f"Best k: {best_k}")

# Plot k values vs cross-validation scores
plt.plot(k_range, cv_scores)
plt.xlabel('k')
plt.ylabel('Cross-Validated Accuracy')
plt.title('K value vs Accuracy')
plt.show()

# Table form
results_df_accuracy = pd.DataFrame({
    'k': list(k_range),
    'Accuracy': cv_scores
})

print(results_df_accuracy)
# Best k is 10
#%%
# Choosing best k (based on 10-fold CV, f1_weighted)

cv_scores = []

k_range = range(1, 20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
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
# Choosing best k (based on Stratified 10-fold CV, accuracy)

strat_kfold = StratifiedKFold(n_splits=10)

# List to store cross-validation F1 scores
cv_accuracy = []

# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='accuracy')
    cv_accuracy.append(accuracy.mean())

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'Accuracy': cv_accuracy
})

print(results_df_accuracy)
# %%
# Choosing best k (based on Stratified 10-fold CV, f1_weighted)

# Stratified k-fold to ensure each fold has a similar class distribution
strat_kfold = StratifiedKFold(n_splits=10)

# List to store cross-validation F1 scores
cv_f1_scores = []

# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    f1_scores = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='f1_weighted')
    cv_f1_scores.append(f1_scores.mean())

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'F1 Score (weighted)': cv_f1_scores
})

print(results_df_f1)
#%%
# Run knn with optimal k
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(X_train, y_train))
print('kNN accuracy for test set: %f' % knn_model.score(X_test, y_test))
