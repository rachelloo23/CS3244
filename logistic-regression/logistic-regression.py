# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
logisticregression_model = LogisticRegression(multi_class='multinomial', solver='lbfgs',random_state=31 ,max_iter=1000)
logisticregression_model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1_train = f1_score(y_train, model.predict(X_train))
print("f1_train:", f1_train)
#Accuracy_train: 0.9924037594953007
f1 = f1_score(y_test, y_pred)
print("f1 score:", accuracy)
#Accuracy: 0.9459203036053131

print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support
#
#           1       0.94      0.99      0.97       496
#           2       0.96      0.94      0.95       471
#           3       0.99      0.96      0.98       420
#           4       0.97      0.88      0.92       508
#           5       0.90      0.97      0.94       556
#           6       1.00      1.00      1.00       545
#           7       0.89      0.70      0.78        23
#           8       1.00      1.00      1.00        10
#           9       0.63      0.69      0.66        32
#          10       0.79      0.76      0.78        25
#          11       0.68      0.65      0.67        49
#          12       0.71      0.63      0.67        27
#
#    accuracy                           0.95      3162
#   macro avg       0.87      0.85      0.86      3162
#weighted avg       0.95      0.95      0.95      3162
