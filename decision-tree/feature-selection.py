# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import random
# %%
subject_id = pd.read_csv("../data/Train/subject_id_train.txt", header=None)
X_train =  pd.read_csv("../data/Train/X_train.txt", header=None, delim_whitespace=True)
y_train = pd.read_csv("../data/Train/y_train.txt", header=None)
X_test = pd.read_csv("../data/Test/X_test.txt", header=None, delim_whitespace=True)
y_test = pd.read_csv("../data/Test/y_test.txt", header=None)
features = pd.read_csv("../data/features.txt", header=None)
# %%
def create_df(X_train, subject_id, y_train, features):
    # Create a copy of the X_train dataframe
    train = X_train.copy()

    # Add "id" and "label" columns
    train["id"] = subject_id.iloc[:, 0]
    train["label"] = y_train.iloc[:, 0]

    # Use feature names from 'features'
    feature_names = features.iloc[:, 0].values
    train.columns = np.append(feature_names, ["id", "label"])

    # Display the created dataframe
    display(train)

    # Display the value counts of 'label' where id == 1
    display(train[train["id"] == 1]["label"].value_counts())

    return train
train = create_df(X_train, subject_id, y_train, features)
test = create_df(X_test, subject_id, y_test, features)
# %%
train.to_csv("../data/processed/train.csv", index=False)
test.to_csv("../data/processed/test.csv", index=False)
# %%
# Calculate the correlation matrix for the training set
correlation_matrix = X_train.corr().abs()
# Create a mask for the upper triangle of the correlation matrix
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
# Find the index of feature columns that have a correlation greater than 0.8
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
# Drop the features from both the training and testing set
X_train_selected = X_train.drop(columns=to_drop)
X_test_selected = X_test.drop(columns=to_drop)
# Display the results
print("Dropped features:", to_drop)
print("Original Dataframe shape: ", X_train.shape)
print("Reduced DataFrame shape:", X_train_selected.shape)
#Dropped features: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 26, 27, 30, 34, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 66, 67, 68, 70, 71, 72, 74, 75, 76, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 109, 110, 113, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 146, 150, 151, 154, 155, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 189, 191, 193, 194, 200, 201, 202, 203, 205, 206, 207, 208, 210, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 236, 239, 240, 241, 242, 243, 244, 245, 246, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 262, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 297, 299, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 372, 373, 374, 376, 378, 380, 381, 382, 383, 384, 385, 386, 387, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 455, 457, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 507, 508, 509, 510, 514, 515, 516, 517, 518, 520, 521, 522, 523, 527, 528, 529, 530, 531, 533, 534, 535, 536, 538, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 553, 558, 559, 560]
#Original Dataframe shape:  (17076, 561)
#Reduced DataFrame shape: (17076, 145)

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

clf = DecisionTreeClassifier(random_state=31)
decision_tree_model = clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 score:", f1)
# f1 score: 0.7146145134668216
# %%
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', decision_tree_model )])
# 1. Using cross_val_score
scores = cross_val_score(pipeline, X_train_selected, y_train, cv=10, scoring=make_scorer(f1_score, average='micro'))

print("Base model scores: ", scores)
# Base model scores:  [0.86357786 0.76962677 0.81338481 0.76576577 0.84813385 0.80952381
# 0.85585586 0.90850515 0.88530928 0.8556701 ]

print("Average f1 score: ", scores.mean())
#Average f1 score:  0.8375353261951201

# %%
# 2. Using RandomizedSearchCV
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', decision_tree_model)])
# Define hyperparameter space for RandomizedSearchCV
param_dist = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': np.arange(2, 10),
    'classifier__min_samples_leaf': np.arange(1, 10),
    'classifier__max_features': [None, 'sqrt', 'log2'],
}
# Define RandomizedSearchCV with 10-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings that are sampled
    scoring=make_scorer(f1_score, average='micro'),
    cv=10,  # 10-fold cross-validation for each parameter setting
    random_state=31,
    n_jobs=-1  # Use all available cores
)
# Fit RandomizedSearchCV to the training data
random_search.fit(X_train_selected, y_train)
# Get the best parameters
print("Best hyperparameters found: ", random_search.best_params_)
# Predict using the best model
y_pred = random_search.best_estimator_.predict(X_test_selected)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# Best hyperparameters found:  {'classifier__min_samples_split': np.int64(5), 'classifier__min_samples_leaf': np.int64(9), 'classifier__max_features': None, 'classifier__max_depth': 40, 'classifier__criterion': 'entropy'}
#              precision    recall  f1-score   support
#
#           1       0.84      0.93      0.88       496
#           2       0.83      0.81      0.82       471
#           3       0.85      0.79      0.82       420
#           4       0.83      0.80      0.81       508
#           5       0.82      0.88      0.85       556
#           6       1.00      1.00      1.00       545
#           7       0.50      0.70      0.58        23
#           8       0.75      0.30      0.43        10
#           9       0.47      0.53      0.50        32
#          10       0.47      0.32      0.38        25
#          11       0.65      0.27      0.38        49
#          12       0.50      0.59      0.54        27
#
#    accuracy                           0.85      3162
#   macro avg       0.71      0.66      0.67      3162
#weighted avg       0.85      0.85      0.85      3162
# %%
misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test.values.ravel(), y_pred)) if true != pred]
print("Misclassified instances:")
for i in misclassified_indices:
    print(f"Index: {i}, True label: {y_test.iloc[i, 0]}, Predicted label: {y_pred[i]}, Features: {X_test.iloc[i].values}")

print("Misclassified labels and index:")
for i in misclassified_indices:
    print(f"Index: {i}, True label: {y_test.iloc[i, 0]}, Predicted label: {y_pred[i]}")
# %%