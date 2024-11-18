# Gradient Boosting Model for Activity Recognition

This folder contains the implementation of a Gradient Boosting (XGBoost) model for activity recognition using tabular data. The project involves data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation.

## Folder Structure

```
.
├── config/
│   └── config.yaml             # Contains best model hyperparameters
├── log/                        # Tuning logs (not uploaded)
├── model/
│   └── model.pkl               # Saved trained model
├── results/
│   ├── defFeat_8.txt           # Default params, features with corr ≥ 0.8
│   ├── defFeat_9.txt           # Default params, features with corr ≥ 0.9
│   ├── defFeatSmote_8.txt      # Default params, features with corr ≥ 0.8, SMOTE oversampling
│   ├── defFeatSmote_9.txt      # Default params, features with corr ≥ 0.9, SMOTE oversampling
│   ├── defOrig.txt             # Default params, original data
│   ├── defOrigSmote.txt        # Default params, original data, SMOTE oversampling
│   ├── tuned_FeatSmote_8.txt   # Tuned params, features with corr ≥ 0.8, SMOTE oversampling
│   ├── xgb_tune_results.csv    # Hyperparameter tuning results
│   ├── xgb_tune_results_2.csv  # Additional tuning results
│   └── xgb_tune_results_3.csv  # Additional tuning results
├── main.py                     # Main script for model training and evaluation
└── tune.py                     # Script for hyperparameter tuning
```

## Key Components

	•	Data Preprocessing and Feature Selection:
	•	Features are selected based on correlation thresholds (≥ 0.8 and ≥ 0.9) to reduce multicollinearity.
	•	SMOTE (Synthetic Minority Over-sampling Technique) is applied to address class imbalance in the dataset.
	•	Model Architecture (main.py and tune.py):
	•	main.py: The main script used for training the model with default or tuned hyperparameters and evaluating its performance.
	•	tune.py: Contains the hyperparameter tuning process using Ray Tune and Optuna to find the optimal hyperparameters for the XGBoost model.
	•	Hyperparameter Tuning:
	•	Utilizes Ray Tune and Optuna for efficient hyperparameter optimization.
	•	Explores various combinations of hyperparameters such as:
	•	Number of estimators (trees).
	•	Maximum depth of trees.
	•	Learning rate.
	•	Subsample ratios.
	•	Regularization parameters.
	•	The best hyperparameters are saved in config/config.yaml.
	•	Model Evaluation:
	•	Performance is evaluated using metrics like accuracy, precision, recall, and F1-score.
	•	Confusion matrices and classification reports are generated to assess the model’s performance on different classes.
	•	Results are saved in the results/ directory.



