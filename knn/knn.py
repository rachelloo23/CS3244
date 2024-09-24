# %%
import csv
import math
import random
import numpy as np
import pandas as pd
# %%

# Let's read the data in as a "data frame" (df), equivalent to our D = (X,y) data matrix
train = pd.read_csv('C:/Users/Xin Rong/OneDrive - National University of Singapore/Desktop/CS3244/Group Project/CS3244_repo/data/processed/train.csv',sep=',') 
test = pd.read_csv('C:/Users/Xin Rong/OneDrive - National University of Singapore/Desktop/CS3244/Group Project/CS3244_repo/data/processed/test.csv',sep=',') 


X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]
# print(X_test.columns)

print(y_test.count())
#%%
# Proportion of labels in train and test are roughly the same (1% diff)
for col in y_test.columns:
    print(f"Proportions for column: {col}")
    print(y_test[col].value_counts(normalize=True))
    print()

# %%

# Get the machine learning algorithm k-NN
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors = 1, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(X_train, y_train))
print('kNN accuracy for test set: %f' % knn_model.score(X_test, y_test))
#%%

# Choosing best k (based on accuracy)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


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
# Choosing best k (based on f1_weighted)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


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

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

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
strat_kfold = StratifiedKFold(n_splits=10)

# List to store cross-validation F1 scores
cv_f1_scores = []

# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    f1_scores = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='accuracy')
    cv_f1_scores.append(f1_scores.mean())

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'Accuracy': cv_f1_scores
})

print(results_df_f1)
#%%

############################################################################################################

# confusion matrix
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix for the test set
cm = confusion_matrix(y_test, y_test_pred)

# Visualizing the confusion matrix as a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Test Set')
plt.show()
#%%

import numpy as np
from collections import Counter

# Calculate the misclassification flag
test_errors = y_test != y_test_pred

# Count misclassifications by true label
error_counts = Counter(y_test[test_errors])

# Count the total number of instances for each label in y_test
total_counts = Counter(y_test)

# Calculate error rates for each class
error_rates = {label: error_counts.get(label, 0) / total_counts[label] for label in total_counts}

# Display error rates
print("Class-wise error rates:", error_rates)

# Visualize class-wise error rates
plt.bar(error_rates.keys(), error_rates.values())
plt.xlabel('Class')
plt.ylabel('Error Rate')
plt.title('Class-wise Error Rates')
plt.show()
