# %%
import csv
import math
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
# Misclassification rates for each label

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
# Normalize the confusion matrix row-wise
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Find the most misclassified classes
misclassified_rates = pd.DataFrame({
    "True Class": range(1, len(conf_matrix) + 1),
    "Misclassified Rate": [1 - normalized_conf_matrix[i, i] for i in range(len(conf_matrix))]
}).sort_values(by="Misclassified Rate", ascending=False)

print("Most Misclassified Classes:")
print(misclassified_rates)
# Most Misclassified Classes:
#     True Class  Misclassified Rate
# 10          11            0.489796
# 11          12            0.481481
# 2            3            0.204762
# 9           10            0.200000
# 6            7            0.173913
# 8            9            0.156250
# 3            4            0.110236
# 4            5            0.102518
# 1            2            0.099788
# 0            1            0.024194
# 5            6            0.005505
# 7            8            0.000000