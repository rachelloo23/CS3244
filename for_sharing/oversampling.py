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

print(classification_report(y_test, knn_model.predict(X_train)))
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

#%%

# Step 1: Import necessary libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Step 3: Apply LDA
lda = LDA(n_components=None)  # Leave n_components=None to maximize class separability
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Step 4: Train a Linear SVM model on LDA-transformed data
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train_lda, y_train)

# Step 5: Predict on the test set
y_pred_train = linear_svm.predict(X_train_lda)
y_pred_test = linear_svm.predict(X_test_lda)

# Step 6: Evaluate the model
print("Training Evaluation:")
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train)}")
print(f"Training F1 Score (weighted): {f1_score(y_train, y_pred_train, average='weighted')}")

print("\nTest Evaluation:")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")
print(f"Test F1 Score (weighted): {f1_score(y_test, y_pred_test, average='weighted')}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
