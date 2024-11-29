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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE

clf = DecisionTreeClassifier(random_state=31)
decision_tree_model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# %%
pipeline = Pipeline([
    ('smote', SMOTE(random_state=31)),               # SMOTE for handling class imbalance
    ('scaler', StandardScaler()),                    # Standardization
    ('model', decision_tree_model)                                     # Decision Tree model
])
# Perform cross-validation to evaluate the model
scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='f1_micro')  # Change scoring as needed

# Print the average F1 score from cross-validation
print("Average F1 Score (with SMOTE and Standardization):", np.mean(scores))
# Average F1 Score (with SMOTE and Standardization): 0.8126885722246546

# Fit the pipeline on the training data and evaluate on the test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_train,pipeline.predict(X_train) , digits=5))
print(classification_report(y_test, y_pred, digits=5))
# precision    recall  f1-score   support

#           1       0.77      0.89      0.82       496
#           2       0.80      0.72      0.76       471
#           3       0.84      0.79      0.81       420
#           4       0.86      0.78      0.82       508
#           5       0.81      0.87      0.84       556
#           6       0.99      0.98      0.99       545
#           7       0.58      0.61      0.60        23
#           8       1.00      0.90      0.95        10
#           9       0.55      0.56      0.55        32
#          10       0.59      0.76      0.67        25
#          11       0.58      0.57      0.58        49
#          12       0.67      0.59      0.63        27
#
#    accuracy                           0.83      3162
#   macro avg       0.75      0.75      0.75      3162
#weighted avg       0.83      0.83      0.83      3162
# %%
misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test.values.ravel(), y_pred)) if true != pred]
print("Misclassified instances:")
for i in misclassified_indices:
    print(f"Index: {i}, True label: {y_test.iloc[i, 0]}, Predicted label: {y_pred[i]}, Features: {X_test.iloc[i].values}")

print("Misclassified labels and index:")
for i in misclassified_indices:
    print(f"Index: {i}, True label: {y_test.iloc[i, 0]}, Predicted label: {y_pred[i]}")
# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap = "OrRd_r")
plt.title('Confusion Matrix for Decision Tree Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print("Confusion Matrix:")
print(cm)
# %%
