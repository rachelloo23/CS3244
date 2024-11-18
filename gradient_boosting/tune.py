# %%
import numpy as np
import xgboost as xgb
import random
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from xgboost import XGBClassifier
from main import writeResults
from imblearn.over_sampling import SMOTE
# %%
#Set random seed
random_seed = 31
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
# %%
train_8 = pd.read_csv("../data/processed/train_8.csv")
test_8 = pd.read_csv("../data/processed/test_8.csv")
print(train_8.head())
print(test_8.head())
# %%
# Split the data into X (features) and y (labels)
X_train = train_8.drop(["label"], axis=1)
y_train = train_8["label"]
y_train = y_train - 1
y_test = test_8["label"] - 1
X_test = test_8.drop(["label"], axis=1)
smote = SMOTE(random_state=random_seed)
X_train_smote_8, y_train_smote_8 = smote.fit_resample(X_train, y_train)
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
        objective='multi:softmax',
        eval_metric='mlogloss',
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
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', model)])
    scores = cross_val_score(pipeline, X_train_smote_8, y_train_smote_8, cv=10, scoring=make_scorer(f1_score, average='micro'))
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
    run_config=RunConfig(storage_path=os.path.abspath("log"), name="xgb_trial_3", log_to_file=True)
)

# %%
xgb_results = analysis_xgb.fit()
# %%
xgb_df = xgb_results.get_dataframe()
best_result_xgb = xgb_results.get_best_result("f1", mode="max")
best_config_xgb = best_result_xgb.config

# %%
print("Best hyperparameters for XGBClassifier: ", best_config_xgb)
print("Best F1 score for XGBClassifier: ", best_result_xgb)
xgb_df.to_csv("xgb_tune_results_3.csv", index=False)
# Shutdown Ray
ray.shutdown()
# %%
