# %%
import csv
import math
import random
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
# %%
# Reading in data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] 

X_test = test.iloc[:, :-2]
y_test = test[['label']]

# Get the machine learning algorithm k-NN 

knn = neighbors.KNeighborsClassifier(n_neighbors = 7, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(X_train, y_train))
print('kNN accuracy for test set: %f' % knn_model.score(X_test, y_test))
# kNN accuracy for training set: 0.970388
# kNN accuracy for test set: 0.892473

print(f1_score(y_train, knn_model.predict(X_train) , average='weighted')) # 0.9700585744061878
print(f1_score(y_test, knn_model.predict(X_test), average='weighted')) # 0.8905021655360718
#%%
# Choosing best k (based on 10-fold CV, accuracy)

cv_scores = []

k_range = range(1, 20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Perform 10-fold cross-validation and take the mean accuracy
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Determine the best k (k with the highest accuracy)
best_k = k_range[cv_scores.index(max(cv_scores))]

print(f"Best k: {best_k}")

# Plot k values vs cross-validation scores
plt.plot(k_range, cv_scores)
plt.xlabel('k')
plt.ylabel('Cross-Validated Accuracy')
plt.title('K value vs Accuracy')
plt.show()

# Table form
results_df_accuracy = pd.DataFrame({
    'k': list(k_range),
    'Accuracy': cv_scores
})

print(results_df_accuracy)
# Best k is 10
#%%
# Choosing best k (based on 10-fold CV, f1_weighted)

cv_scores = []

k_range = range(1, 20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Perform 10-fold cross-validation and take the weighted f1
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='f1_weighted')
    # For multiclass:
    # 'f1_macro': Averages the F1 score for each class without considering class imbalance.
    # 'f1_weighted': Averages the F1 score for each class, weighted by the number of samples in each class.
    cv_scores.append(scores.mean())

# Determine the best k (k with the highest Weighted f1)
best_k = k_range[cv_scores.index(max(cv_scores))]

print(f"Best k: {best_k}")

# Plot k values vs cross-validation scores
plt.plot(k_range, cv_scores)
plt.xlabel('k')
plt.ylabel('Cross-Validated Weighted f1')
plt.title('K value vs Weighted f1')
plt.show()

# Table form
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'F1 Score': cv_scores
})

print(results_df_f1)
# Best k is 10

#%%
# Choosing best k (based on Stratified 10-fold CV, accuracy)

strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=31)

# List to store cross-validation F1 scores
cv_accuracy = []

k_range = range(1, 20)
# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='accuracy')
    cv_accuracy.append(accuracy.mean())

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'Accuracy': cv_accuracy
})

# Determine the best k (k with the highest Weighted f1)
best_k = k_range[cv_accuracy.index(max(cv_accuracy))]

print(f"Best k: {best_k}")

# Best k is 10
# %%
# Choosing best k (based on Stratified 10-fold CV, f1_weighted)
random_seed = 31
# Stratified k-fold to ensure each fold has a similar class distribution
strat_kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state=random_seed)
# strat_kfold = StratifiedKFold(n_splits=10)
# List to store cross-validation F1 scores
cv_f1_scores = []
temp_1 = []
k_range = range(1, 20)

# Loop through different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,  metric='euclidean')
    f1_scores = cross_val_score(knn, X_train, y_train, cv=strat_kfold, scoring='f1_weighted')
    temp_1.append(f1_scores)
    cv_f1_scores.append(f1_scores.mean())

# Create a DataFrame to store k values and corresponding F1 scores
results_df_f1 = pd.DataFrame({
    'k': list(k_range),
    'F1 Score (weighted)': cv_f1_scores
})
# Determine the best k (k with the highest Weighted f1)
best_k = k_range[cv_f1_scores.index(max(cv_f1_scores))]

print(f"Best k: {best_k}")
print(results_df_f1)
print(temp_1)
# Best k is 5
#%%
# Run knn with optimal k (k = 10)
knn = neighbors.KNeighborsClassifier(n_neighbors = 5, metric='euclidean')
knn_model = knn.fit(X_train, y_train) 

print('kNN accuracy for training set: %f' % knn_model.score(X_train, y_train))
print('kNN accuracy for test set: %f' % knn_model.score(X_test, y_test))
#%%
# Using Stratified K fold CV, F1_weighted score, with random seed

# k = 1
# kNN accuracy for training set: 1.000000
# kNN accuracy for test set: 0.864643

# k = 5
# kNN accuracy for training set: 0.976053
# kNN accuracy for test set: 0.884883

# Best k: 1
#      k  F1 Score (weighted)
# 0    1             0.958114
# 1    2             0.942332
# 2    3             0.953863
# 3    4             0.949061
# 4    5             0.951912
# 5    6             0.949532
# 6    7             0.950242
# 7    8             0.947719
# 8    9             0.948877
# 9   10             0.946637
# 10  11             0.947668
# 11  12             0.945145
# 12  13             0.945008
# 13  14             0.944650
# 14  15             0.942737
# 15  16             0.943755
# 16  17             0.942193
# 17  18             0.943071
# 18  19             0.939495

# Breakdown of F1_weighted scores for each k (1 to 20) -> at most 1% difference

# [array([0.95421812, 0.96134794, 0.94866187, 0.95967418, 0.96615467,
#        0.96771147, 0.94817399, 0.95359777, 0.95800798, 0.96358975]), array([0.92962808, 0.95415797, 0.94464016, 0.93702308, 0.94472725,
#        0.94668696, 0.93948412, 0.93857446, 0.93161707, 0.95677731]), array([0.9461514 , 0.9580834 , 0.94082033, 0.95279815, 0.95816278,
#        0.96784245, 0.95039357, 0.95677291, 0.95349776, 0.95410435]), array([0.94884716, 0.94835492, 0.93909623, 0.94787456, 0.94778654,
#        0.96115315, 0.94933891, 0.94308722, 0.94162754, 0.96344672]), array([0.9527935 , 0.9515982 , 0.94160586, 0.94279273, 0.95510693,
#        0.95750844, 0.94649907, 0.95681291, 0.95732955, 0.95707491]), array([0.94506835, 0.94771559, 0.94689085, 0.94239555, 0.95124487,
#        0.95988495, 0.94086695, 0.94909987, 0.95105352, 0.9610954 ]), array([0.95027518, 0.94896225, 0.94289881, 0.94248427, 0.95425139,
#        0.95588078, 0.94098063, 0.95176552, 0.95656248, 0.95836357]), array([0.9567599 , 0.94358162, 0.93648756, 0.94225627, 0.95486275,
#        0.95614171, 0.9399413 , 0.93996277, 0.94874212, 0.95844991]), array([0.95002576, 0.95170133, 0.93532952, 0.93954516, 0.95461055,
#        0.9610532 , 0.93556473, 0.95038561, 0.95462   , 0.95593622]), array([0.95198005, 0.94137307, 0.92899186, 0.93791011, 0.95105199,
#        0.95962974, 0.9373707 , 0.94656883, 0.95308839, 0.95840223]), array([0.94846372, 0.95151133, 0.93276147, 0.93783055, 0.95099951,
#        0.95703859, 0.93985852, 0.94760971, 0.9547111 , 0.9558905 ]), array([0.9494919 , 0.94612218, 0.93658348, 0.93260224, 0.94441618,
#        0.95309317, 0.93739817, 0.94767418, 0.95210307, 0.95196404]), array([0.94446613, 0.94479147, 0.93399626, 0.9354443 , 0.94710312,
#        0.95810199, 0.94129235, 0.94346128, 0.95211428, 0.94931255]), array([0.94306546, 0.94353946, 0.93281365, 0.93550818, 0.94123151,
#        0.95427774, 0.93838629, 0.94724978, 0.95853232, 0.95189477]), array([0.94223619, 0.94994112, 0.93729692, 0.92751039, 0.93755528,
#        0.95144941, 0.93409412, 0.94339571, 0.95334557, 0.95054208]), array([0.94317891, 0.9479254 , 0.93653425, 0.93273172, 0.94149325,
#        0.95136536, 0.93386189, 0.94757973, 0.95646944, 0.94641319]), array([0.93962973, 0.94917986, 0.93905252, 0.93530247, 0.93889142,
#        0.94606904, 0.9327262 , 0.94074123, 0.95122824, 0.94910663]), array([0.93444335, 0.94658848, 0.93795322, 0.9281367 , 0.94705795,
#        0.95253446, 0.93252294, 0.94493616, 0.95516976, 0.95136988]), array([0.93481259, 0.9461023 , 0.93247634, 0.93113499, 0.94255286,
#        0.94590849, 0.92866891, 0.93946645, 0.95184623, 0.94198145])]

#%%
# Using Stratified K fold CV, F1_weighted score, without random seed, best: k = 10
# kNN accuracy for training set: 0.965109
# kNN accuracy for test set: 0.888362

# Best k: 10
#      k  F1 Score (weighted)
# 0    1             0.876504
# 1    2             0.863506
# 2    3             0.889018
# 3    4             0.888338
# 4    5             0.893212
# 5    6             0.894501
# 6    7             0.894313
# 7    8             0.894813
# 8    9             0.894745
# 9   10             0.894868
# 10  11             0.894248
# 11  12             0.894583
# 12  13             0.893839
# 13  14             0.892745
# 14  15             0.891731
# 15  16             0.893309
# 16  17             0.891044
# 17  18             0.892513
# 18  19             0.890254

# Breakdown of F1_weighted scores for each k (1 to 20) -> more significant diff

# [array([0.88131527, 0.80676029, 0.89670435, 0.85685726, 0.87323605,
#        0.85533212, 0.88968918, 0.90247111, 0.8957142 , 0.90695864]), array([0.86517595, 0.7861457 , 0.8799263 , 0.8569308 , 0.86696325,
#        0.85991707, 0.87764724, 0.88685298, 0.87590093, 0.8796004 ]), array([0.9060341 , 0.80748554, 0.9050952 , 0.86271324, 0.8862646 ,
#        0.86763546, 0.90411403, 0.91854803, 0.90099644, 0.93128987]), array([0.90487123, 0.81594643, 0.89744276, 0.87373988, 0.88865381,
#        0.8712029 , 0.9068523 , 0.91387469, 0.88996459, 0.92083273]), array([0.92188393, 0.82811438, 0.89234437, 0.86725151, 0.89230201,
#        0.86745338, 0.91546808, 0.91645729, 0.89946206, 0.93137977]), array([0.92185796, 0.82752769, 0.89342063, 0.86976897, 0.89277073,
#        0.8671166 , 0.91733264, 0.92026718, 0.90479374, 0.93014951]), array([0.93351945, 0.82206621, 0.90012971, 0.86808586, 0.89148662,
#        0.86415237, 0.91633791, 0.91360806, 0.90769245, 0.92605221]), array([0.92705916, 0.82290807, 0.90166452, 0.86725416, 0.89825305,
#        0.86463105, 0.91978528, 0.91346362, 0.90327092, 0.92984168]), array([0.93768635, 0.81784522, 0.8987247 , 0.86220893, 0.89514041,
#        0.86697569, 0.91866995, 0.91536243, 0.89977858, 0.93505754]), array([0.9225383 , 0.81933954, 0.89953545, 0.87219835, 0.89336927,
#        0.86295669, 0.91978132, 0.91980414, 0.90404461, 0.9351148 ]), array([0.92823118, 0.82455349, 0.89757019, 0.86358522, 0.89515474,
#        0.85625685, 0.9148148 , 0.91830524, 0.90895177, 0.93505893]), array([0.92747686, 0.82848551, 0.88768695, 0.86302351, 0.89835368,
#        0.86343125, 0.91862992, 0.91942954, 0.90807427, 0.9312416 ]), array([0.92428035, 0.81993802, 0.89751589, 0.8652678 , 0.8961316 ,
#        0.8590707 , 0.91221428, 0.9193964 , 0.90821554, 0.93635897]), array([0.93100587, 0.82063819, 0.89303806, 0.86620752, 0.8943567 ,
#        0.86075187, 0.90744724, 0.92075107, 0.90720352, 0.92604574]), array([0.92928642, 0.82235963, 0.89258389, 0.85423651, 0.89525486,
#        0.8536288 , 0.90867002, 0.92204668, 0.90805446, 0.93118807]), array([0.92427984, 0.82964877, 0.90257558, 0.86616952, 0.88910708,
#        0.85546102, 0.91244993, 0.92093227, 0.90803388, 0.92443115]), array([0.92676088, 0.82859694, 0.89655202, 0.85920163, 0.88874097,
#        0.84946747, 0.91508782, 0.91942517, 0.90222482, 0.92437753]), array([0.92553597, 0.82264334, 0.90331212, 0.86374984, 0.88932972,
#        0.85380838, 0.91623647, 0.91832097, 0.90640856, 0.92578837]), array([0.92651746, 0.81858749, 0.90194949, 0.86478633, 0.88522077,
#        0.8450711 , 0.91122107, 0.92229062, 0.90460642, 0.92228643])]
