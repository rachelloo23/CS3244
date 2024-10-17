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
from sklearn.metrics import f1_score, classification_report
# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]
#%% 
# Distribution of data points in training only
import matplotlib.pyplot as plt

# Assuming y_train is a pandas Series or a numpy array
(unique_labels, label_counts) = np.unique(y_train, return_counts=True)

# Create a bar plot for label frequencies
plt.figure(figsize=(8, 6))
plt.bar(unique_labels, label_counts, color='skyblue', edgecolor='black')

# Add title and labels
plt.title('Frequency Distribution of y_train Labels', fontsize=15)
plt.xlabel('Class Label', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Display the plot
plt.show()
import numpy as np

# Assuming y_train is a numpy array
(unique_labels, label_counts) = np.unique(y_train, return_counts=True)

# Print each label and its corresponding count
for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: {count} occurrences")

#%%

# Oversampling using random over sampler

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Apply RandomOverSampler to oversample the minority class
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)


knn = neighbors.KNeighborsClassifier(n_neighbors = 7, metric='euclidean')
knn_model = knn.fit(X_train_ros, y_train_ros) 


print(f1_score(y_train_ros, knn_model.predict(X_train_ros) , average='weighted')) # 0.9896637821765382
print(f1_score(y_test, knn_model.predict(X_test), average='weighted')) # 0.8921237480429026

print(classification_report(y_test, knn_model.predict(X_test)))
#               precision    recall  f1-score   support

#            1       0.86      0.98      0.92       496
#            2       0.88      0.91      0.90       471
#            3       0.96      0.79      0.86       420
#            4       0.90      0.81      0.85       508
#            5       0.85      0.92      0.88       556
#            6       1.00      0.99      0.99       545
#            7       0.86      0.78      0.82        23
#            8       0.67      1.00      0.80        10
#            9       0.63      0.84      0.72        32
#           10       0.62      0.80      0.70        25
#           11       0.76      0.65      0.70        49
#           12       0.75      0.44      0.56        27

#     accuracy                           0.89      3162
#    macro avg       0.81      0.83      0.81      3162
# weighted avg       0.90      0.89      0.89      3162

#%%

# Use SMOTE (can use cause all data is numerical)
# Import necessary libraries
from imblearn.over_sampling import SMOTE
from collections import Counter

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

knn = neighbors.KNeighborsClassifier(n_neighbors = 7, metric='euclidean')
knn_model = knn.fit(X_train_smote, y_train_smote) 


print(f1_score(y_train_smote, knn_model.predict(X_train_smote) , average='weighted')) # 0.9891925159538429
print(f1_score(y_test, knn_model.predict(X_test), average='weighted')) # 0.8948994998711476
print(classification_report(y_test, knn_model.predict(X_test)))
#               precision    recall  f1-score   support

#            1       0.90      0.97      0.93       496
#            2       0.88      0.90      0.89       471
#            3       0.92      0.82      0.87       420
#            4       0.87      0.86      0.87       508
#            5       0.88      0.88      0.88       556
#            6       1.00      0.98      0.99       545
#            7       0.56      0.83      0.67        23
#            8       0.71      1.00      0.83        10
#            9       0.62      0.88      0.73        32
#           10       0.61      0.76      0.68        25
#           11       0.76      0.63      0.69        49
#           12       0.67      0.44      0.53        27

#     accuracy                           0.90      3162
#    macro avg       0.78      0.83      0.80      3162
# weighted avg       0.90      0.90      0.89      3162


