import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
train_df = pd.read_csv('/Users/rachelloo/Downloads/Y4S1/CS3244/CS3244_Project/data/processed/train.csv',sep=",")
test_df = pd.read_csv('/Users/rachelloo/Downloads/Y4S1/CS3244/CS3244_Project/data/processed/test.csv',sep=",")
features = pd.read_csv('/Users/rachelloo/Downloads/Y4S1/CS3244/CS3244_Project/data/features.txt',sep=",")
X_train = train_df.drop(columns =['id','label'])
y_train  = train_df['label']
X_test = test_df.drop(columns =['id','label'])
y_test = test_df['label']

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy_train:", accuracy_train)
#Accuracy_train: 0.9924037594953007
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#Accuracy: 0.9459203036053131

print(classification_report(y_test, y_pred))

#precision    recall  f1-score   support
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
