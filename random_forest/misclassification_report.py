import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

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

# Save up to 1000 misclassified instances for each class to a CSV
misclassified_sampled_df = (
    misclassified_df.groupby('actual_label')
    .apply(lambda x: x.sample(n=min(len(x), 1000), random_state=42))
    .reset_index(drop=True)
)
# Drop the 'label' column, replaced by actual_label column which minus 1 to to all values in column for zero based indexing
misclassified_df = misclassified_df.drop(columns=['label'])

# Trim leading and trailing spaces from all column names
misclassified_df.columns = misclassified_df.columns.str.strip()
# Reorder columns: move 'label', 'actual_label', and 'predicted_label' to the beginning
cols = ['actual_label', 'predicted_label'] + [
    col for col in misclassified_df.columns if col not in ['actual_label', 'predicted_label']
]
misclassified_df = misclassified_df[cols]

# Save the misclassified instances to a CSV file
misclassified_df.to_csv("misclassified_instances.csv", index=False)

print("Saved misclassified instances to 'misclassified_instances.csv'")
