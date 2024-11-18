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
print(classification_report(y_test, y_pred))
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
# Misclassified instances:
#Index: 17, True label: 4, Predicted label: 5, Features: [ 4.67666638e-02 -4.25429914e-04 -3.94582187e-02 -9.87815840e-01
# -8.85041593e-01 -9.57315721e-01 -9.87589896e-01 -8.99341030e-01
# -9.63037434e-01 -7.79182401e-01 -6.07351255e-01 -6.89452587e-01
#  8.39739284e-01  6.74468728e-01  6.43144427e-01 -9.51434626e-01
# -9.99818812e-01 -9.95131408e-01 -9.98590413e-01 -9.87410212e-01
# -9.26221629e-01 -9.68585437e-01 -3.74040321e-01 -3.92213233e-01
# -6.81048684e-01  3.64322240e-01 -3.79378631e-01  6.19008442e-01
# -8.37176994e-01  8.07849009e-02 -1.75705451e-01  1.07777986e-01
#  1.94245062e-01  3.60795010e-01 -3.09285304e-01  7.18113962e-02
#  1.63619749e-01  1.51538256e-01 -2.95321808e-01  4.50390324e-01
#  9.48670845e-01 -1.22339985e-01  1.73437641e-01 -9.79429395e-01
# -8.84560524e-01 -9.56492992e-01 -9.79258180e-01 -8.97616490e-01
# -9.57902105e-01  8.84651253e-01 -1.31096676e-01  1.75631662e-01
#  9.65094104e-01 -1.57457052e-01  1.51338810e-01 -2.78613942e-01
#  8.62011305e-01 -9.75942787e-01 -9.44233608e-01 -9.78643691e-01
# -9.35705768e-01 -9.60408549e-01 -1.00000000e+00 -1.00000000e+00
# -1.24805935e-01 -4.86215960e-01  4.84110406e-01 -4.82225225e-01
#  4.80569683e-01 -8.88358680e-01  8.87907591e-01 -8.87633795e-01
#  8.86964295e-01 -8.95440275e-01  8.98134910e-01 -9.00604207e-01
#  9.01808437e-01  6.77833801e-01 -2.39746421e-02  7.18813658e-01
#  7.41676242e-02 -8.84295665e-02 -3.85522045e-02 -9.87661480e-01
# -9.50622663e-01 -9.80487511e-01 -9.89832041e-01 -9.54875713e-01
# -9.79228138e-01 -9.79618443e-01 -9.66611470e-01 -9.78504953e-01
#  9.86672796e-01  9.33142761e-01  9.78097237e-01 -9.78005000e-01
#...
#Index: 3157, True label: 2, Predicted label: 1
#Index: 3158, True label: 2, Predicted label: 1
#Index: 3159, True label: 2, Predicted label: 1
#Index: 3160, True label: 2, Predicted label: 1
