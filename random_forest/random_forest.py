# %%
import numpy as np
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

# %%
# Set random seed
random_seed = 31
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

# %%
# Load the train and test datasets
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

# Drop the "id" column as it's not needed for modeling
train = train.drop(["id"], axis=1)
test = test.drop(["id"], axis=1)

# Split the data into X (features) and y (labels)
X_train = train.drop(["label"], axis=1)
y_train = train["label"] - 1  # Adjusting label for zero-indexing
y_test = test["label"] - 1
X_test = test.drop(["label"], axis=1)

# %%
# Define the search space for Random Forest hyperparameters
param_dist_rf = {
    "n_estimators": tune.randint(50, 300),              # Number of trees
    "max_depth": tune.randint(3, 30),                   # Maximum depth of the tree
    "max_features": tune.choice(["sqrt", "log2", None]), # Max number of features considered for splitting
    "min_samples_split": tune.randint(2, 10),           # Minimum samples required to split an internal node
    "min_samples_leaf": tune.randint(1, 4),             # Minimum samples required to be at a leaf node
    "bootstrap": tune.choice([True, False])             # Whether bootstrap samples are used when building trees
}

# %%
# Define the objective function for Random Forest tuning
def objective_rf(config):
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        max_features=config["max_features"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        bootstrap=config["bootstrap"],
        random_state=random_seed
    )
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring=make_scorer(f1_score, average='micro'))
    session.report({'f1': scores.mean()})

# %%
# Run the hyperparameter tuning for Random Forest
ray.shutdown()  # Shutdown Ray if it was already running
ray.init()  # Initialize Ray

analysis_rf = tune.Tuner(
    objective_rf,
    tune_config=tune.TuneConfig(
        metric="f1",
        mode="max",
        scheduler=ASHAScheduler(),
        num_samples=50,  # Try 50 different configurations
    ),
    param_space=param_dist_rf,
    run_config=RunConfig(storage_path=os.path.abspath("log_rf"), name="rf_trial_1", log_to_file=True)
)

rf_results = analysis_rf.fit()

# %%
# Get the best hyperparameters from the tuning process
rf_df = rf_results.get_dataframe()
best_result_rf = rf_results.get_best_result("f1", mode="max")
best_config_rf = best_result_rf.config

# Print the best hyperparameters
print("Best hyperparameters for RandomForestClassifier: ", best_config_rf)

# %%
# Train the final Random Forest model with the best hyperparameters
rf_tuned = RandomForestClassifier(
    n_estimators=best_config_rf['n_estimators'],
    max_depth=best_config_rf['max_depth'],
    max_features=best_config_rf['max_features'],
    min_samples_split=best_config_rf['min_samples_split'],
    min_samples_leaf=best_config_rf['min_samples_leaf'],
    bootstrap=best_config_rf['bootstrap'],
    random_state=random_seed
)
rf_tuned.fit(X_train, y_train)

# %%
# Evaluate the model on training and test data
print('Random Forest - Training set score: {:.4f}'.format(rf_tuned.score(X_train, y_train)))
print('Random Forest - Test set score: {:.4f}'.format(rf_tuned.score(X_test, y_test)))

# %%
# Predict on the test set
y_test_pred_rf = rf_tuned.predict(X_test)

# Compute the confusion matrix
cm_rf = confusion_matrix(y_test, y_test_pred_rf)

print('Random Forest Confusion matrix\n\n', cm_rf)

# Print additional classification metrics
print('\nRandom Forest Classification Report\n')
print(classification_report(y_test, y_test_pred_rf))

# %%
# Save the tuning results
rf_df.to_csv("rf_results.csv", index=False)

# Shutdown Ray after completion
ray.shutdown()

# Best hyperparameters for RandomForestClassifier:  {'n_estimators': 171, 'max_depth': 19, 'max_features': 'log2', 'min_samples_split': 7, 'min_samples_leaf': 1, 'bootstrap': False}
# Random Forest - Training set score: 1.0000
# Random Forest - Test set score: 0.9222
# Random Forest Confusion matrix

#  [[477  10   9   0   0   0   0   0   0   0   0   0]
#  [ 30 434   7   0   0   0   0   0   0   0   0   0]
#  [ 18  46 356   0   0   0   0   0   0   0   0   0]
#  [  0   0   0 451  55   0   1   1   0   0   0   0]
#  [  0   0   0  13 543   0   0   0   0   0   0   0]
#  [  0   1   0   0   0 544   0   0   0   0   0   0]
#  [  1   1   0   2   0   0  17   1   0   0   1   0]
#  [  0   0   0   0   0   0   1   9   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  24   0   8   0]
#  [  0   0   0   0   0   0   0   0   0  19   1   5]
#  [  3   0   0   1   0   1   3   0  11   0  30   0]
#  [  1   1   0   0   0   0   0   0   0  11   2  12]]

# Random Forest Classification Report

#               precision    recall  f1-score   support

#            0       0.90      0.96      0.93       496
#            1       0.88      0.92      0.90       471
#            2       0.96      0.85      0.90       420
#            3       0.97      0.89      0.93       508
#            4       0.91      0.98      0.94       556
#            5       1.00      1.00      1.00       545
#            6       0.77      0.74      0.76        23
#            7       0.82      0.90      0.86        10
#            8       0.69      0.75      0.72        32
#            9       0.63      0.76      0.69        25
#           10       0.71      0.61      0.66        49
#           11       0.71      0.44      0.55        27

#     accuracy                           0.92      3162
#    macro avg       0.83      0.82      0.82      3162
# weighted avg       0.92      0.92      0.92      3162