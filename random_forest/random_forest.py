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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# %%
# Set random seed
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

# Load the train and test datasets
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

# Drop the "id" column as it's not needed for modeling
train = train.drop(["id"], axis=1)
test = test.drop(["id"], axis=1)

# Split the data into X (features) and y (labels)
X_train = train.drop(["label"], axis=1)
y_train = train["label"] - 1  # Adjusting label for zero-indexing
X_test = test.drop(["label"], axis=1)
y_test = test["label"] - 1

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=random_seed)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define the search space for Random Forest hyperparameters
param_dist_rf = {
    "n_estimators": tune.randint(50, 300),
    "max_depth": tune.randint(3, 30),
    "max_features": tune.choice(["sqrt", "log2", None]),
    "min_samples_split": tune.randint(2, 10),
    "min_samples_leaf": tune.randint(1, 4),
    "bootstrap": tune.choice([True, False])
}

# Define the objective function for Random Forest tuning
def objective_rf(config):
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        max_features=config["max_features"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        bootstrap=config["bootstrap"],
        random_state=42  # Use a fixed seed for reproducibility
    )
    
    # Define stratified K-fold with standardization within each fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in skf.split(X_train, y_train):
        # Split into training and validation sets
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Standardize each fold independently
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        # Train the model on the current fold
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate the model on the validation fold
        val_pred = model.predict(X_val_fold)
        f1 = f1_score(y_val_fold, val_pred, average='macro')
        scores.append(f1)
    
    # Report the average F1 score across all folds
    session.report({'f1': sum(scores) / len(scores)})

# Initialize Ray and run the hyperparameter tuning
ray.shutdown()  # Shutdown Ray if it was already running
ray.init()  # Initialize Ray

analysis_rf = tune.Tuner(
    objective_rf,
    tune_config=tune.TuneConfig(
        metric="f1",
        mode="max",
        scheduler=ASHAScheduler(),
        num_samples=20
    ),
    param_space=param_dist_rf,
    run_config=RunConfig(storage_path=os.path.abspath("log_rf"), name="rf_trial_1", log_to_file=True)
)

rf_results = analysis_rf.fit()

# Get the best hyperparameters from the tuning process
rf_df = rf_results.get_dataframe()
best_result_rf = rf_results.get_best_result("f1", mode="max")
best_config_rf = best_result_rf.config

print("Best hyperparameters for RandomForestClassifier: ", best_config_rf)

# Train the final Random Forest model with the best hyperparameters
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_tuned = RandomForestClassifier(
    n_estimators=best_config_rf['n_estimators'],
    max_depth=best_config_rf['max_depth'],
    max_features=best_config_rf['max_features'],
    min_samples_split=best_config_rf['min_samples_split'],
    min_samples_leaf=best_config_rf['min_samples_leaf'],
    bootstrap=best_config_rf['bootstrap'],
    random_state=42
)
rf_tuned.fit(X_train_scaled, y_train)

# Evaluate the model on training and test data
train_score = rf_tuned.score(X_train_scaled, y_train)
test_score = rf_tuned.score(X_test_scaled, y_test)

print('Random Forest - Training set score: {:.4f}'.format(train_score))
print('Random Forest - Test set score: {:.4f}'.format(test_score))

# Predict on the test set
y_test_pred_rf = rf_tuned.predict(X_test_scaled)

# Compute the confusion matrix
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
print('Random Forest Confusion matrix\n\n', cm_rf)

# Print additional classification metrics
print('\nRandom Forest Classification Report\n')
print(classification_report(y_test, y_test_pred_rf))

# Save the performance results to a text file
results_path = os.path.join(os.getcwd(), "base_model_results.txt")

with open(results_path, "w") as results_file:
    results_file.write("Random Forest Model Performance Metrics\n")
    results_file.write("=" * 50 + "\n\n")
    results_file.write(f"Best Hyperparameters: {best_config_rf}\n\n")
    results_file.write(f"Training Set Score: {train_score:.4f}\n")
    results_file.write(f"Test Set Score: {test_score:.4f}\n\n")
    results_file.write("Confusion Matrix:\n")
    results_file.write(str(cm_rf) + "\n\n")
    results_file.write("Classification Report:\n")
    results_file.write(classification_report + "\n")
# Save the tuning results
rf_df.to_csv("rf_results.csv", index=False)

# Shutdown Ray after completion
ray.shutdown()

###### Without feature selection #########
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



###### After feature selection and oversampling #########
# Best hyperparameters for RandomForestClassifier:  {'n_estimators': 152, 'max_depth': 22, 'max_features': 'sqrt', 'min_samples_split': 8, 'min_samples_leaf': 3, 'bootstrap': False}
# Random Forest - Training set score: 0.9999
# Random Forest - Test set score: 0.9152
# Random Forest Confusion matrix

#  [[480   8   8   0   0   0   0   0   0   0   0   0]
#  [ 34 429   8   0   0   0   0   0   0   0   0   0]
#  [ 17  48 355   0   0   0   0   0   0   0   0   0]
#  [  0   0   0 454  53   0   1   0   0   0   0   0]
#  [  0   0   0  38 518   0   0   0   0   0   0   0]
#  [  0   1   0   0   0 544   0   0   0   0   0   0]
#  [  0   1   0   2   0   0  18   1   1   0   0   0]
#  [  0   0   0   0   0   0   1   9   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  26   0   6   0]
#  [  0   0   0   0   0   0   0   0   0  17   1   7]
#  [  3   1   0   1   0   0   3   0  13   0  28   0]
#  [  0   0   0   0   0   0   0   0   0   8   3  16]]

# Random Forest Classification Report

#               precision    recall  f1-score   support

#            0       0.90      0.97      0.93       496
#            1       0.88      0.91      0.89       471
#            2       0.96      0.85      0.90       420
#            3       0.92      0.89      0.91       508
#            4       0.91      0.93      0.92       556
#            5       1.00      1.00      1.00       545
#            6       0.78      0.78      0.78        23
#            7       0.90      0.90      0.90        10
#            8       0.65      0.81      0.72        32
#            9       0.68      0.68      0.68        25
#           10       0.74      0.57      0.64        49
#           11       0.70      0.59      0.64        27

#     accuracy                           0.92      3162
#    macro avg       0.83      0.82      0.83      3162
# weighted avg       0.92      0.92      0.91      3162

