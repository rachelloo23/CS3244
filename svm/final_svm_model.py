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

#%%

model = LinearSVC(C=0.1, max_iter=5000, random_state=random_seed) # tuned params
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)

f1 = f1_score(y_train_smote, model.predict(X_train_smote), average='micro')  
print("F1 Score on train Set:", f1)
f1 = f1_score(y_test, y_pred, average='micro')  
print("F1 Score on Test Set:", f1)

# F1 Score on train Set: 0.9958421175919419
# F1 Score on Test Set: 0.9456040480708412