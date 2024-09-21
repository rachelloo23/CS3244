# %%
import numpy as np
import xgboost as xgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, confusion_matrix, classification_report, make_scorer, r2_score
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from ray.air import session
# %%
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
print(train.head())
print(test.head())
# %%
train = train.drop(["id"], axis=1)
test = test.drop(["id"], axis=1)
print(train.head())
print(test.head())
# %%
# Split the data into X (features) and y (labels)
X_train = train.drop(["label"], axis=1)
y_train = train["label"]
y_train = y_train - 1
y_test = test["label"] - 1
X_test = test.drop(["label"], axis=1)

# %%
param_dist_xgb = {
    "n_estimators": tune.randint(10, 30),
    "max_depth": tune.randint(3, 15),
    "learning_rate": tune.loguniform(0.001, 0.2),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.6, 1.0),
    "gamma": tune.loguniform(1e-8, 1.0),
    "reg_alpha": tune.loguniform(1e-8, 1.0),
    "reg_lambda": tune.loguniform(1e-8, 1.0)
}
# %%
# Define the XGBoost model
ray.shutdown()
ray.init()
def objective_xgb(config):
    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        gamma=config["gamma"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        random_state=123
    )
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring=make_scorer(f1_score, average='micro'))
    # model.fit(latent_train_z, latent_train_y)
    # y_val_pred = model.predict(latent_val_z)
    # val_f1 = f1_score(latent_val_y, y_val_pred)
    session.report({'f1':scores.mean()})
# %%
# Run the hyperparameter search for XGBClassifier
analysis_xgb = tune.Tuner(
    objective_xgb,
    tune_config=tune.TuneConfig(
    metric="f1",
    mode="max",
    scheduler=ASHAScheduler(),
    num_samples=10,
    ),
    param_space=param_dist_xgb,
)

# %%
xgb_results = analysis_xgb.fit()
# %%
xgb_df = xgb_results.get_dataframe()
best_result_xgb = xgb_results.get_best_result("f1", mode="max")
best_config_xgb = best_result_xgb.config

# %%
print("Best hyperparameters for XGBClassifier: ", best_config_xgb)

# Train the final models with the best hyperparameters

xgb_tuned = XGBClassifier(
    n_estimators=best_config_xgb['n_estimators'],
    max_depth=best_config_xgb['max_depth'],
    learning_rate=best_config_xgb['learning_rate'],
    subsample=best_config_xgb['subsample'],
    colsample_bytree=best_config_xgb['colsample_bytree'],
    gamma=best_config_xgb['gamma'],
    reg_alpha=best_config_xgb['reg_alpha'],
    reg_lambda=best_config_xgb['reg_lambda']
)
xgb_tuned.fit(X_train, y_train)

# Print the scores on the training and validation set
print('XGBoost - Training set score: {:.4f}'.format(xgb_tuned.score(X_train, y_train)))
print('XGBoost - Test set score: {:.4f}'.format(xgb_tuned.score(X_test, y_test)))
# %%
# Predict on the test set
y_test_pred_xgb = xgb_tuned.predict(X_test)

# Compute the confusion matrix
cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)

print('XGBoost Confusion matrix\n\n', cm_xgb)

# Print additional classification metrics
print('\nXGBoost Classification Report\n')
print(classification_report(y_test, y_test_pred_xgb))

# Shutdown Ray
ray.shutdown()

# %%
xgb_df.to_csv("xgb_results.csv", index=False)

# %%
# Best hyperparameters for XGBClassifier:  {'n_estimators': 25, 'max_depth': 3, 'learning_rate': 0.11382600922866078, 'subsample': 0.8887835193955859, 'colsample_bytree': 0.621090024617741, 'gamma': 0.0004938098621938259, 'reg_alpha': 2.7397283095559356e-06, 'reg_lambda': 2.00618138131514e-07}
# XGBoost - Training set score: 0.9674
# XGBoost - Test set score: 0.8969
# XGBoost Confusion matrix

#  [[485   2   9   0   0   0   0   0   0   0   0   0]
#  [ 46 416   9   0   0   0   0   0   0   0   0   0]
#  [  9  49 362   0   0   0   0   0   0   0   0   0]
#  [  0   1   0 408  98   0   0   1   0   0   0   0]
#  [  0   0   0  50 506   0   0   0   0   0   0   0]
#  [  0   0   0   0   0 545   0   0   0   0   0   0]
#  [  2   2   1   1   0   0  17   0   0   0   0   0]
#  [  0   0   0   0   0   0   0  10   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  24   0   8   0]
#  [  0   0   0   0   0   0   0   1   0  15   0   9]
#  [  2   0   0   2   1   0   1   1   9   0  33   0]
#  [  1   1   0   0   0   0   1   0   0   8   1  15]]

# XGBoost Classification Report

#               precision    recall  f1-score   support

#            0       0.89      0.98      0.93       496
#            1       0.88      0.88      0.88       471
#            2       0.95      0.86      0.90       420
#            3       0.89      0.80      0.84       508
#            4       0.84      0.91      0.87       556
#            5       1.00      1.00      1.00       545
#            6       0.89      0.74      0.81        23
#            7       0.77      1.00      0.87        10
#            8       0.73      0.75      0.74        32
#            9       0.65      0.60      0.63        25
#           10       0.79      0.67      0.73        49
#           11       0.62      0.56      0.59        27

#     accuracy                           0.90      3162
#    macro avg       0.82      0.81      0.82      3162
# weighted avg       0.90      0.90      0.90      3162
