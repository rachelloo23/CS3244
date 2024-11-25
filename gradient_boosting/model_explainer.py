# %%
import numpy as np
import xgboost as xgb
import random
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import yaml
from imblearn.over_sampling import SMOTE
from module import *
from lime.lime_tabular import LimeTabularExplainer
# %%
#Set random seed
random_seed = 31
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
# %%
# Load the best hyperparameters for XGBClassifier
CONFIG_PATH = "config/"
config = load_config(CONFIG_PATH, "config.yaml")
print(config)
# %%
train_8 = pd.read_csv("../data/processed/train_8.csv")
test_8 = pd.read_csv("../data/processed/test_8.csv")

print(train_8.head())
print(test_8.head())
# %%
X_train_8 = train_8.drop(["label"], axis=1)
y_train_8 = train_8["label"]
y_train_8 = y_train_8 - 1
y_test_8 = test_8["label"] - 1
X_test_8 = test_8.drop(["label"], axis=1)
# %%
# Standardize the features
scaler = StandardScaler()
X_train_8 = scaler.fit_transform(X_train_8)
X_test_8 = scaler.transform(X_test_8)
# %%
# Train the final models with the best hyperparameters
smote = SMOTE(random_state=random_seed)
X_train_smote_8, y_train_smote_8 = smote.fit_resample(X_train_8, y_train_8)
xgb_tuned = XGBClassifier(
    n_estimators=config['n_estimators'],
    max_depth=config['max_depth'],
    learning_rate=config['learning_rate'],
    subsample=config['subsample'],
    colsample_bytree=config['colsample_bytree'],
    gamma=config['gamma'],
    reg_alpha=config['reg_alpha'],
    reg_lambda=config['reg_lambda'],
    random_state=random_seed
)
xgb_tuned.fit(X_train_smote_8, y_train_smote_8)
# %%
# Highest misclassification rate for non-transition classes
# Class 3: Misclassification Rate = 0.0417
# Highest misclassification rate for transition classes
# Class 10: Misclassification Rate = 0.0082
res = misclass_analysis(y_test_8, X_test_8, 3, xgb_tuned)
print(res)
res = misclass_analysis(y_test_8, X_test_8, 10, xgb_tuned)
print(res)
# %%
import matplotlib.pyplot as plt
import numpy as np

# Select indices to explain
indices_to_explain = [17, 18]

# Define feature names and class names based on your dataset
feature_names = train_8.drop(["label"], axis=1).columns.tolist()
class_names = list(range(len(np.unique(y_train_8))))  # Assuming classes are 0, 1, ..., n-1

# Create a LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train_smote_8,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification"
)

# Dictionary to store explanations for each instance
explanations = {}

# Generate explanations for each index
for idx in indices_to_explain:
    # Reshape the instance for prediction
    instance = X_test_8[idx].reshape(1, -1)
    
    # Generate the explanation
    explanation = explainer.explain_instance(
        data_row=instance.flatten(),
        predict_fn=xgb_tuned.predict_proba
    )
    
    # Extract the feature contributions
    explanations[idx] = dict(explanation.as_list())

# Combine all features from both instances
all_features = list(set().union(*[explanations[idx].keys() for idx in indices_to_explain]))

# Prepare data for plotting
values_17 = [explanations[17].get(feature, 0) for feature in all_features]
values_50 = [explanations[18].get(feature, 0) for feature in all_features]

# Sort features by the sum of contributions for better visualization
sorted_indices = np.argsort([abs(v17) + abs(v50) for v17, v50 in zip(values_17, values_50)])[::-1]
sorted_features = [all_features[i] for i in sorted_indices]
values_17 = [values_17[i] for i in sorted_indices]
values_50 = [values_50[i] for i in sorted_indices]

# Plot the contributions with stacking
plt.figure(figsize=(12, 8))
positions = np.arange(len(sorted_features))

# Add the bars for each instance
plt.barh(
    positions, values_17, height=0.6, label="Instance 17", color="green", alpha=0.7, edgecolor="black"
)
plt.barh(
    positions, values_50, height=0.6, label="Instance 50", color="red", alpha=0.7, edgecolor="black", left=values_17
)

# Formatting
plt.yticks(positions, sorted_features)
plt.xlabel("Feature Importance")
plt.title("LIME Explanation: Stacked Feature Contributions for Instance 17 and 50")
plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig("lime_feature_importance_stacked.png")
plt.show()
# %%
