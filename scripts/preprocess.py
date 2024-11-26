# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
subject_id_train = pd.read_csv("../data/Train/subject_id_train.txt", header=None)
subject_id_test = pd.read_csv("../data/Test/subject_id_test.txt", header=None)
X_train =  pd.read_csv("../data/Train/X_train.txt", header=None, delim_whitespace=True)
y_train = pd.read_csv("../data/Train/y_train.txt", header=None)
X_test = pd.read_csv("../data/Test/X_test.txt", header=None, delim_whitespace=True)
y_test = pd.read_csv("../data/Test/y_test.txt", header=None)
features = pd.read_csv("../data/features.txt", header=None)
# %%
print(X_train)
print(y_train)
# %%
y_train.iloc[:,0].value_counts().plot(kind="bar")
# plt.savefig('class_dist.png', format='png', dpi=400)
# %%
display(features)
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
train = create_df(X_train, subject_id_train, y_train, features)
test = create_df(X_test, subject_id_test, y_test, features)
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()
# %%
train.to_csv("../data/processed/train.csv", index=False)
test.to_csv("../data/processed/test.csv", index=False)
# %%
