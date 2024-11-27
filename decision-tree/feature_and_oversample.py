# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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

# Feature selection based on correlation
def feature_selection(X_train, correlation_threshold=0.9):
    # Calculate the correlation matrix for the training set
    correlation_matrix = X_train.corr().abs()
    # Create a mask for the upper triangle of the correlation matrix
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    # Find the index of feature columns that have a correlation greater than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    # Drop the features from the training set
    X_train_selected = X_train.drop(columns=to_drop)
    return X_train_selected, to_drop

# Apply feature selection
X_train_selected, to_drop = feature_selection(X_train)
X_test_selected = X_test.drop(columns=to_drop)

# Print feature selection results
print("Dropped features:", to_drop)
print("Original DataFrame shape: ", X_train.shape)
print("Reduced DataFrame shape:", X_train_selected.shape)

# Define the model pipeline with feature selection, oversampling, and scaling
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=31)),         # Oversampling
    ('scaler', StandardScaler()),              # Standardization
    ('classifier', DecisionTreeClassifier(random_state=31))  # Decision Tree model
])

# Define hyperparameter space for RandomizedSearchCV
param_dist = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': np.arange(2, 10),
    'classifier__min_samples_leaf': np.arange(1, 10),
    'classifier__max_features': [None, 'sqrt', 'log2'],
}

# Use RandomizedSearchCV for hyperparameter tuning
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

# Evaluation metrics
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred, digits = 5))
#Dropped features: [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 59, 60, 61, 66, 67, 68, 70, 71, 72, 74, 75, 76, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 109, 113, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 146, 154, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 189, 193, 200, 201, 202, 203, 205, 206, 207, 208, 210, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 231, 232, 233, 234, 239, 240, 241, 242, 244, 245, 246, 249, 252, 253, 254, 255, 257, 258, 259, 260, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 297, 299, 301, 302, 303, 310, 311, 312, 313, 314, 315, 316, 317, 318, 324, 325, 326, 327, 328, 329, 330, 332, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 372, 374, 376, 378, 380, 381, 382, 383, 384, 385, 386, 389, 390, 391, 392, 393, 394, 396, 397, 398, 399, 400, 403, 404, 405, 406, 407, 408, 410, 411, 412, 413, 414, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 455, 457, 459, 460, 461, 462, 467, 468, 469, 470, 471, 472, 473, 474, 476, 482, 483, 484, 485, 486, 487, 488, 490, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 507, 508, 509, 510, 514, 515, 516, 517, 518, 520, 521, 522, 523, 527, 528, 529, 530, 531, 533, 534, 535, 536, 540, 541, 542, 543, 544, 546, 547, 548, 549, 553, 558, 559, 560]
#Original DataFrame shape:  (7767, 561)
#Reduced DataFrame shape: (7767, 212)
#Best hyperparameters found:  {'classifier__min_samples_split': np.int64(8), 'classifier__min_samples_leaf': np.int64(9), 'classifier__max_features': None, 'classifier__max_depth': 10, 'classifier__criterion': 'entropy'}
#Classification Report on Test Set:
#              precision    recall  f1-score   support
#           1       0.83      0.88      0.85       496
#           2       0.81      0.78      0.80       471
#           3       0.84      0.82      0.83       420
#           4       0.83      0.80      0.81       508
#           5       0.82      0.83      0.83       556
#           6       0.99      0.99      0.99       545
#           7       0.48      0.57      0.52        23
#           8       0.62      0.80      0.70        10
#           9       0.73      0.69      0.71        32
#          10       0.56      0.60      0.58        25
#          11       0.67      0.71      0.69        49
#          12       0.54      0.48      0.51        27
#    accuracy                           0.84      3162
#   macro avg       0.73      0.75      0.73      3162
#weighted avg       0.84      0.84      0.84      3162


# %%
