# Gradient Boosting Model for Activity Recognition

This repository contains the implementation of a Gradient Boosting (XGBoost) model for activity recognition using tabular data. Below is a detailed guide on the files, workflow, and how to reproduce the experiments.

## Files

### Python Scripts
- **`main.py`**: The main script used for training the XGBoost model with default or tuned hyperparameters and evaluating its performance.
- **`tune.py`**: Script for hyperparameter tuning using Ray Tune and Optuna to find the optimal hyperparameters for the XGBoost model.
- **`model_explainer.py`**: Script to analyze and explain the model's predictions using LIME.

### Configuration
- **`config/config.yaml`**: Contains the best model hyperparameters determined during the tuning process.

### Data Files
- **`results/xgb_tune_results_4.csv`**, **`xgb_tune_results_5.csv`**: CSV files containing hyperparameter tuning results.
- **`results/defFeat_8.txt`**, **`defFeat_9.txt`**, **`defFeatSmote_8.txt`**, etc.: Text files documenting the results for various experimental setups, including feature selection thresholds and oversampling methods.

### Visualizations
- **`lime_feature_importance_comparison_class_4_vs_3.png`**: A bar chart comparing feature importances for specific classes using LIME.

## Reproducing Results

### Step 1: Train and Evaluate the Model
Run `main.py` to train the model and evaluate its performance:
```bash
python main.py
```

### Step 2: Tune Hyperparameters
Run tune.py to perform hyperparameter tuning using Ray Tune and Optuna:
```bash
python tune.py
```
The best hyperparameters will be saved in `config/config.yaml`.

### Step 3: Analyze Model Predictions
Run model_explainer.py to analyze the model’s predictions and generate LIME explanations:
```bash
python model_explainer.py
```
