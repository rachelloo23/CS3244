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
print("Classification Report on Train Set:")
print(classification_report(y_train, random_search.best_estimator_.predict(X_train_selected), digits = 5))
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred, digits = 5))

# %%
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
# Normalize the confusion matrix row-wise
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Create a heatmap for visualization
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(1, 13), yticklabels=range(1, 13))
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Find the most misclassified classes
misclassified_rates = pd.DataFrame({
    "True Class": range(1, len(conf_matrix) + 1),
    "Misclassified Rate": [1 - normalized_conf_matrix[i, i] for i in range(len(conf_matrix))]
}).sort_values(by="Misclassified Rate", ascending=False)

print("Most Misclassified Classes:")
print(misclassified_rates)

# %%
