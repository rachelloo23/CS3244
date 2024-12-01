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

## name of string of file to save
model_name = "base_model_results.txt"
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

# Define the search space for Random Forest hyperparameters
param_dist_rf = {
    "n_estimators": tune.randint(50, 150),  
    "max_depth": tune.randint(3, 15),  
    "max_features": tune.choice(["sqrt", "log2"]),  
    "min_samples_split": tune.randint(2, 5),  
    "min_samples_leaf": tune.randint(1, 3),  
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
        num_samples=10
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

# Ensure the folder exists
results_folder = "Model_results"
os.makedirs(results_folder, exist_ok=True)

# Save the performance results to a text file
results_path = os.path.join(results_folder, model_name)

with open(results_path, "w") as results_file:
    results_file.write("Random Forest Model Performance Metrics\n")
    results_file.write("=" * 50 + "\n\n")
    results_file.write(f"Best Hyperparameters: {best_config_rf}\n\n")
    results_file.write(f"Training Set Score: {train_score:.4f}\n")
    results_file.write(f"Test Set Score: {test_score:.4f}\n\n")
    results_file.write("Confusion Matrix:\n")
    results_file.write(str(cm_rf) + "\n\n")
    results_file.write("Classification Report:\n")
    # Fix the issue here by calling the classification_report function and writing the output
    results_file.write(classification_report(y_test, y_test_pred_rf) + "\n")


# Ensure the folder exists
folder_path = "log_rf"
os.makedirs(folder_path, exist_ok=True)

# Save the tuning results in the folder
file_path = os.path.join(folder_path, "base_model_tuning.csv")
rf_df.to_csv(file_path, index=False)

# Shutdown Ray after completion
ray.shutdown()