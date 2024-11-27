import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

import random
import os

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Load train and test data
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

# Drop the 'id' column (if present)
train = train.drop(["id"], axis=1, errors='ignore')
test = test.drop(["id"], axis=1, errors='ignore')

# Split data into X (features) and y (labels)
X_train = train.drop(["label"], axis=1)
y_train = train["label"] - 1  # Adjust for zero-based indexing

X_test = test.drop(["label"], axis=1)
y_test = test["label"] - 1  # Adjust for zero-based indexing

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the model with best hyperparameters
rf_tuned = RandomForestClassifier(
    n_estimators=152,
    max_depth=22,
    max_features='sqrt',
    min_samples_split=8,
    min_samples_leaf=3,
    bootstrap=False,
    random_state=42
)

# Fit the model on the training data
rf_tuned.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
print(f'Train Accuracy: {rf_tuned.score(X_train_scaled, y_train):.4f}')
print(f'Test Accuracy: {rf_tuned.score(X_test_scaled, y_test):.4f}')

# Predict on the test set
y_test_pred = rf_tuned.predict(X_test_scaled)

# Compute and display confusion matrix and classification report
cm = confusion_matrix(y_test, y_test_pred)
print('Confusion Matrix:\n', cm)
print('\nClassification Report:\n', classification_report(y_test, y_test_pred))

# Identify misclassified instances
misclassified_indices = np.where(y_test != y_test_pred)[0]  # Indices of misclassified rows
misclassified_df = test.iloc[misclassified_indices].copy()  # Extract those rows

# Add columns for actual and predicted labels
misclassified_df['actual_label'] = y_test.iloc[misclassified_indices].values
misclassified_df['predicted_label'] = y_test_pred[misclassified_indices]

# Choose one misclassified instance
misclassified_index = misclassified_indices[0]  # First misclassified instance
instance = X_test_scaled[misclassified_index]  # Feature values of the selected instance
actual_label = y_test.iloc[misclassified_index]  # True label
predicted_label = y_test_pred[misclassified_index]  # Predicted label

# Initialize LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X_train.columns.tolist(),
    class_names=[f'class_{i}' for i in range(len(np.unique(y_train)))],
    mode='classification'
)


# Generate explanation for the misclassified instance
explanation = explainer.explain_instance(instance, rf_tuned.predict_proba, num_features=10)

# Handle predicted label mismatch
if predicted_label in explanation.local_exp:
    explanation_list = explanation.as_list(label=predicted_label)
else:
    # Fallback and provide additional context
    fallback_label = list(explanation.local_exp.keys())[0]
    print(
        f"Warning: Predicted label {predicted_label} not found in explanation. "
        f"Using label {fallback_label} instead.\n"
        f"Actual label: {actual_label}, Predicted label: {predicted_label}"
    )
    explanation_list = explanation.as_list(label=fallback_label)

# Plot the feature importance for the misclassified instance
features, weights = zip(*explanation_list)

plt.figure(figsize=(8, 6))
plt.barh(features, weights)
plt.xlabel("Feature Weight", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title(f"LIME Explanation for Misclassified Instance\nActual: {actual_label}, Predicted: {predicted_label}", fontsize=14)
plt.tight_layout()
plt.show()

print("Misclassified Instance Feature Contributions:")
for feature, weight in explanation_list:
    print(f"{feature}: {weight}")

# Function to find a correct instance of a given class
def find_correct_instance(class_label, y_true, y_pred):
    correct_indices = np.where((y_true == class_label) & (y_true == y_pred))[0]
    if len(correct_indices) > 0:
        return X_test_scaled[correct_indices[random.choice(range(len(correct_indices)))]]
    else:
        print(f"No correct instance found for class {class_label}.")
        return None

# Directory to save the plots
output_dir = "lime_explanations"
os.makedirs(output_dir, exist_ok=True)

# Function to generate and save LIME plot
def generate_lime_plot(instance, label, predicted_label, explanation, output_filename):
    # Check if the label exists in the explanation before accessing it
    if label in explanation.local_exp:
        explanation_list = explanation.as_list(label=label)
    else:
        # Fallback if the label doesn't exist in the explanation
        fallback_label = list(explanation.local_exp.keys())[0]  # Fallback to first available label
        print(f"Warning: {label} not found in explanation. Using label {fallback_label} instead.")
        explanation_list = explanation.as_list(label=fallback_label)

    features, weights = zip(*explanation_list)

    plt.figure(figsize=(8, 6))
    plt.barh(features, weights)
    plt.xlabel("Feature Weight", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"LIME Explanation\nActual: {actual_label}, Predicted: {predicted_label}", fontsize=14)
    plt.tight_layout()
    
    # Save the plot as PNG
    plot_path = os.path.join(output_dir, output_filename)
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory
    print(f"Saved plot to {plot_path}")

# Find a correct instance of the predicted class
correct_predicted_instance = find_correct_instance(predicted_label, y_test, y_test_pred)
if correct_predicted_instance is not None:
    correct_predicted_explanation = explainer.explain_instance(correct_predicted_instance, rf_tuned.predict_proba, num_features=10)
    generate_lime_plot(correct_predicted_instance, predicted_label, predicted_label, correct_predicted_explanation, f"correct_predicted_class_{predicted_label}.png")

# Find a correct instance of the actual class
correct_actual_instance = find_correct_instance(actual_label, y_test, y_test_pred)
if correct_actual_instance is not None:
    correct_actual_explanation = explainer.explain_instance(correct_actual_instance, rf_tuned.predict_proba, num_features=10)
    generate_lime_plot(correct_actual_instance, actual_label, predicted_label, correct_actual_explanation, f"correct_actual_class_{actual_label}.png")
