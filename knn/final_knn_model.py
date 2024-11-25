# %%
import csv
import math
import random
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

random_seed = 31
#%%
from imblearn.over_sampling import SMOTE
from collections import Counter
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

train.columns = train.columns.str.replace(' ', '') 
test.columns = test.columns.str.replace(' ', '') 
#%%
# Filter labels 1-6
train = train.loc[train['label'].isin([1,2,3,4,5,6])]
test = test.loc[test['label'].isin([1,2,3,4,5,6])]

#%%

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] -1

X_test = test.iloc[:, :-2]
y_test = test[['label']] -1 
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='manhattan')
knn_model = knn.fit(X_train, y_train) 

print(f1_score(y_train, knn_model.predict(X_train) , average='micro'))  # 0.9918013586319981
print(f1_score(y_test, knn_model.predict(X_test), average='micro'))  # 0.9013282732447818

#%%
# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=random_seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='manhattan')
knn_model = knn.fit(X_train_smote, y_train_smote) 

print(f1_score(y_train_smote, knn_model.predict(X_train_smote) , average='micro'))  # 0.9918013586319981
print(f1_score(y_test, knn_model.predict(X_test), average='micro'))  # 0.9013282732447818
#%%
# confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming y_test contains true labels and y_pred contains predictions
y_pred = knn_model.predict(X_test)  # Replace with your model's prediction method

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
disp.plot(cmap='Blues', values_format='d')

plt.title("Confusion Matrix")
plt.show()
#%%
# Error rates for each label
# 0: 19.9%
# 1: 19.6%
# 2: 6.73%
# 3: 16.98%
# 4: 16.98%
# 5: 16.03%
# 6: 4.49%
# 7: 1.92%
# 8: 6.41%
# 9: 3.21%
# 10: 3.21%
# 11: 1.28%

