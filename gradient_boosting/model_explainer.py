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
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

print(train.head())
print(test.head())
# %%
X_train = train.drop(["label"], axis=1)
y_train = train["label"]
y_train = y_train - 1
y_test = test["label"] - 1
X_test = test.drop(["label"], axis=1)
# %%
# Standardize the features
# %%
# Train the final models with the best hyperparameters
smote = SMOTE(random_state=random_seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)
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
xgb_tuned.fit(X_train_smote, y_train_smote)
# %%
# Highest misclassification rate for non-transition classes
# Class 3: Misclassification Rate = 0.0417
# Highest misclassification rate for transition classes
# Class 10: Misclassification Rate = 0.0082
# %%

# Obtain predictions from the model
predictions = xgb_tuned.predict(X_test)

# Ensure y_test is a NumPy array if it's not already
y_test_array = np.array(y_test)

# Indices where Class 4 is correctly classified
indices_correct_class4 = np.where((y_test_array == 4) & (predictions == 4))[0][:5]

# Indices where Class 8 is correctly classified
indices_correct_class8 = np.where((y_test_array == 8) & (predictions == 8))[0][:5]

# Print the indices
print("Indices of correctly classified Class 4 instances:", indices_correct_class4)
print("Indices of correctly classified Class 8 instances:", indices_correct_class8)# %%
# %%
res = misclass_analysis(y_test, X_test, 3, xgb_tuned)
print(res)
res = misclass_analysis(y_test, X_test, 9, xgb_tuned)
print(res)
# %
y_pred = xgb_tuned.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
# %
# %%
def plot_lime_explanation(
    a, b, A, B, X_train_smote, X_test, y_test, xgb_tuned,
    feature_names, class_names, num_feat=6, random_seed=42,
    font_size=14  # New parameter to control font size
):
    """
    Generate a LIME explanation plot comparing feature importances between two instances:
    - Instance 'a': correctly classified as class A
    - Instance 'b': misclassified as class A (true class B)
    
    Parameters:
    - a (int): Index of the instance correctly classified as Class A.
    - b (int): Index of the instance misclassified as Class A (true Class B).
    - A (int or str): Label of Class A.
    - B (int or str): Label of Class B.
    - X_train_smote (array-like): Training data used to fit the LIME explainer.
    - X_test (array-like): Test data.
    - y_test (array-like): True labels for the test data.
    - xgb_tuned (model): Trained model with a predict_proba method.
    - feature_names (list): List of feature names.
    - class_names (list): List of class names.
    - num_feat (int): Number of top features to plot (default is 6).
    - random_seed (int): Random seed for reproducibility (default is 42).
    - font_size (int): Font size for all text elements in the plot (default is 14).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from lime.lime_tabular import LimeTabularExplainer

    # Set the random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Create a LIME explainer with the random_state parameter
    explainer = LimeTabularExplainer(
        training_data=X_train_smote,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=random_seed
    )

    # Dictionary to store explanations for each instance
    explanations = {}

    indices_to_explain = [a, b]

    # Generate explanations for each index
    for idx in indices_to_explain:
        # Reshape the instance for prediction
        instance = X_test[idx].reshape(1, -1)

        # Generate the explanation
        explanation = explainer.explain_instance(
            data_row=instance.flatten(),
            predict_fn=xgb_tuned.predict_proba,
            num_features=num_feat,
            num_samples=1000,       # Fixed number of samples
        )

        # Extract the feature contributions
        explanations[idx] = dict(explanation.as_list())

    # Combine all features from both instances
    all_features = list(set().union(*[explanations[idx].keys() for idx in indices_to_explain]))

    # Prepare data for plotting
    values_a = [explanations[a].get(feature, 0) for feature in all_features]
    values_b = [explanations[b].get(feature, 0) for feature in all_features]

    # Sort features by the sum of contributions for better visualization
    sorted_indices = np.argsort([abs(va) + abs(vb) for va, vb in zip(values_a, values_b)])[::-1]
    sorted_features = [all_features[i] for i in sorted_indices]
    values_a = [values_a[i] for i in sorted_indices]
    values_b = [values_b[i] for i in sorted_indices]

    # Select the number of top features to plot
    features_to_plot = sorted_features[:num_feat]
    values_a_top = [explanations[a].get(feature, 0) for feature in features_to_plot]
    values_b_top = [explanations[b].get(feature, 0) for feature in features_to_plot]

    # Positions for the bars
    positions = np.arange(len(features_to_plot))
    bar_height = 0.35  # Adjust the bar height for better spacing

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot bars for Instance 'a'
    plt.barh(
        positions - bar_height / 2,
        values_a_top,
        height=bar_height,
        color='green',
        alpha=0.5,
        label=f'Correctly Classified as Class {A} (Index {a})'
    )

    # Plot bars for Instance 'b'
    plt.barh(
        positions + bar_height / 2,
        values_b_top,
        height=bar_height,
        color='red',
        alpha=0.5,
        label=f'Misclassified as Class {A} (Index {b}, True Class {B})'
    )

    # Increase font sizes
    plt.yticks(positions, features_to_plot, fontsize=font_size)
    plt.xlabel('Feature Contribution', fontsize=font_size + 2)
    plt.ylabel('Features', fontsize=font_size + 2)
    plt.title(f'LIME Explanation: Class {A} Correct vs Misclassified as Class {A}', fontsize=font_size + 4)
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"lime_feature_importance_comparison_class_{A}_vs_{B}.png")
    plt.show()
# %%
plot_lime_explanation(0, 17, 4, 3, X_train_smote, X_test, y_test, xgb_tuned, feature_names, class_names, num_feat=6, random_seed=random_seed, font_size=14)
# %%
