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
