#%%
import csv
import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]


#%%
svc = SVC()
svc.fit(X_train, y_train)
print(accuracy_score(y_test, svc.predict(X_test))) # 0.9367488931056294
print(accuracy_score(y_train, svc.predict(X_train))) # 0.9716750354062057

print(f1_score(y_test, svc.predict(X_test), average='weighted')) # 0.9360546876940178
print(f1_score(y_train, svc.predict(X_train), average='weighted')) # 0.971551119209504
