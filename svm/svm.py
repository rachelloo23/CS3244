#%%
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC


random_seed = 31

# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]
#%%

# Create the model
model = LinearSVC(C=1.0, max_iter=1000, random_state=random_seed) # default params
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("train f1:",f1_score(y_train, model.predict(X_train), average = 'micro'))
print("test f1:",f1_score(y_test, y_pred, average = 'micro'))
print(classification_report(y_test, y_pred))

# train f1: 0.994850006437492
# test f1: 0.9465528146742568
#               precision    recall  f1-score   support

#            1       0.95      1.00      0.97       496
#            2       0.98      0.95      0.96       471
#            3       1.00      0.98      0.99       420
#            4       0.97      0.87      0.92       508
#            5       0.90      0.98      0.94       556
#            6       1.00      1.00      1.00       545
#            7       0.74      0.61      0.67        23
#            8       0.83      1.00      0.91        10
#            9       0.59      0.59      0.59        32
#           10       0.67      0.72      0.69        25
#           11       0.66      0.63      0.65        49
#           12       0.64      0.52      0.57        27

#     accuracy                           0.95      3162
#    macro avg       0.83      0.82      0.82      3162
# weighted avg       0.95      0.95      0.95      3162

#%%
# Baseline Model (default params, with feature selection, without oversampling)
#######################################################################################################################
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

# Create the model
model = LinearSVC(C=1.0, max_iter=1000, random_state=random_seed) # default params
model.fit(X_train_8, y_train_8)
y_pred = model.predict(X_test_8)

print(f1_score(y_train_8, model.predict(X_train_8) , average='micro'))  # 0.9840350199562251
print(f1_score(y_test_8, model.predict(X_test_8), average='micro'))   # 0.92662871600253

#%%
X_train_9 = train_9.drop('label', axis=1)
y_train_9 = train_9[['label']]

X_test_9 = test_9.drop('label', axis =1)
y_test_9 = test_9[['label']]
model = LinearSVC(C=1.0, max_iter=1000, random_state=random_seed) # default params
model.fit(X_train_9, y_train_9)
y_pred = model.predict(X_test_9)

print(f1_score(y_train_9, model.predict(X_train_9) , average='micro'))  # 0.9864812668984164
print(f1_score(y_test_9, model.predict(X_test_9), average='micro'))   # 0.9294750158127767
#%%

# Baseline Model (default params, without feature selection, with oversampling)

# With SMOTE (can use cause all data is numerical)
from imblearn.over_sampling import SMOTE
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] -1

X_test = test.iloc[:, :-2]
y_test = test[['label']] -1 

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=random_seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
model = LinearSVC(C=1.0, max_iter=1000, random_state=random_seed) # default params
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)

print(f1_score(y_train_smote, model.predict(X_train_smote) , average='micro'))  # 0.9975404075895994
print(f1_score(y_test, model.predict(X_test), average='micro'))   # 0.9468690702087287
#%%

# Baseline Model (default params, with feature selection (0.8), with oversampling)

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=random_seed)
X_train_smote_8, y_train_smote_8 = smote.fit_resample(X_train_8, y_train_8)

model = LinearSVC(C=1.0, max_iter=1000, random_state=random_seed) # default params
model.fit(X_train_8, y_train_8)
y_pred = model.predict(X_test_8)

print(f1_score(y_train_smote_8, model.predict(X_train_smote_8) , average='micro'))  # 0.9911571796673694
print(f1_score(y_test_8, model.predict(X_test_8), average='micro'))   # 0.92662871600253

#%%

# Tune model 
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform

# Create a pipeline with scaling and LinearSVC
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', LinearSVC(random_state=random_seed, max_iter=10000))
])

# Define parameter distribution
param_dist = {
    'svc__C': uniform(0.01, 10),  # Test values between 0.01 and 10
    'svc__loss': ['squared_hinge'],  # Focus on squared hinge for faster tuning
    'svc__dual': [True, False]      # Both dual and primal formulations
}

# Randomized search with fewer iterations
random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=10, cv=10, scoring='f1_micro', n_jobs=-1, random_state=random_seed, verbose=1
)

# Fit the model
random_search.fit(X_train, y_train)

# Display the best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best F1 Score:", random_search.best_score_)
# Best Parameters: {'svc__C': 1.3723134824601035, 'svc__dual': True, 'svc__loss': 'squared_hinge'}
# Best F1 Score: 0.9290746195385371
#%%
# Use the best parameters from RandomizedSearchCV
best_params = random_search.best_params_
best_model = random_search.best_estimator_

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

f1 = f1_score(y_train, best_model.predict(X_train), average='micro')  
print("F1 Score on train Set:", f1)
# F1 Score on train Set: 0.9980687524140595

f1 = f1_score(y_test, y_pred, average='micro')  
print("F1 Score on Test Set:", f1)
# F1 Score on Test Set: 0.9424414927261227
#%%

# Identify misclassified instances (from tuned baseline model)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

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
# Total Misclassified Instances: 182

misclassified_df.to_csv('linearsvc_misclassifications.csv', index=False)
