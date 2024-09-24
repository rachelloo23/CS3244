# %%
import numpy as np
import xgboost as xgb
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, confusion_matrix, classification_report, make_scorer, r2_score
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from ray.air import session
# %%
#Set random seed
random_seed = 31
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
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
    "n_estimators": tune.randint(10, 40),
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
        random_state=random_seed
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
    num_samples=50,
    ),
    param_space=param_dist_xgb,
    run_config=RunConfig(storage_path=os.path.abspath("log"), name="xgb_trial_1", log_to_file=True)
)

# %%
xgb_results = analysis_xgb.fit()
# %%
xgb_df = xgb_results.get_dataframe()
best_result_xgb = xgb_results.get_best_result("f1", mode="max")
best_config_xgb = best_result_xgb.config

# %%
best_config_xgb = {'n_estimators': 37, 'max_depth': 11, 'learning_rate': 0.07743148807027894, 'subsample': 0.7420549200544598, 'colsample_bytree': 0.8999013190460016, 'gamma': 2.335146525850379e-07, 'reg_alpha': 0.005344647514524659, 'reg_lambda': 0.12400234913359398}
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
    reg_lambda=best_config_xgb['reg_lambda'],
    random_state=random_seed
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
# Best hyperparameters for XGBClassifier:  {'n_estimators': 37, 'max_depth': 11, 'learning_rate': 0.07743148807027894, 'subsample': 0.7420549200544598, 'colsample_bytree': 0.8999013190460016, 'gamma': 2.335146525850379e-07, 'reg_alpha': 0.005344647514524659, 'reg_lambda': 0.12400234913359398}
# XGBoost - Training set score: 0.9999
# XGBoost - Test set score: 0.9099
# XGBoost Confusion matrix

#  [[480   4  12   0   0   0   0   0   0   0   0   0]
#  [ 40 426   4   0   0   0   0   0   0   0   1   0]
#  [  8  39 373   0   0   0   0   0   0   0   0   0]
#  [  0   0   0 422  83   0   1   1   1   0   0   0]
#  [  0   0   0  42 514   0   0   0   0   0   0   0]
#  [  0   0   0   0   0 545   0   0   0   0   0   0]
#  [  0   2   0   1   1   0  18   0   0   0   1   0]
#  [  0   0   0   0   0   0   0  10   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  26   0   6   0]
#  [  0   0   0   0   0   0   0   1   0  17   0   7]
#  [  2   0   0   3   0   1   1   1   9   0  32   0]
#  [  1   0   0   0   0   0   0   1   0  10   1  14]]

# XGBoost Classification Report

#               precision    recall  f1-score   support

#            0       0.90      0.97      0.93       496
#            1       0.90      0.90      0.90       471
#            2       0.96      0.89      0.92       420
#            3       0.90      0.83      0.86       508
#            4       0.86      0.92      0.89       556
#            5       1.00      1.00      1.00       545
#            6       0.90      0.78      0.84        23
#            7       0.71      1.00      0.83        10
#            8       0.72      0.81      0.76        32
#            9       0.63      0.68      0.65        25
#           10       0.78      0.65      0.71        49
#           11       0.67      0.52      0.58        27

#     accuracy                           0.91      3162
#    macro avg       0.83      0.83      0.82      3162
# weighted avg       0.91      0.91      0.91      3162
