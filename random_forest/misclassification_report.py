import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

feature_threshold = 0.8

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



#### Adding feature selection
def highCorrFeat(dataframe, threshold):
    """
    Identify highly correlated feature pairs and features to drop based on a given correlation threshold.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing the features.
    threshold (float): The correlation threshold to determine which features are highly correlated. Default is 0.9.
    
    Returns:
    dict: A dictionary of highly correlated feature pairs with their correlation values.
    list: A list of feature columns to drop based on the correlation threshold.
    """
    # Step 1: Calculate the correlation matrix
    correlation_matrix = dataframe.corr().abs()

    # Step 2: Create a mask for the upper triangle
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Step 3: Extract the pairs of highly correlated features
    high_corr_pairs = [(column, row) for column in upper_tri.columns for row in upper_tri.index if upper_tri.loc[row, column] > threshold]

    # Step 4: Store the highly correlated pairs in a dictionary
    res = {}
    for pair in high_corr_pairs:
        corr = correlation_matrix.loc[pair[0], pair[1]]
        res[corr] = [pair[0], pair[1]]

    # Step 5: Find the feature columns that have a correlation greater than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    return res, to_drop
#%%
if feature_threshold == 0.8:
    # Get the features to drop based on threshold 0.8
    res_8, to_drop_8 = highCorrFeat(X_train, 0.8)
    # Drop the features from both the training and testing set
    X_train = X_train.drop(columns=to_drop_8)
    X_test = X_test.drop(columns=to_drop_8)
    y_train = y_train  # If you need to drop columns from target as well (if it includes feature columns)
    print("Using threshold = 0.8")
    print("Dropped features:", to_drop_8)
elif feature_threshold == 0.9:
    # Get the features to drop based on threshold 0.9
    res_9, to_drop_9 = highCorrFeat(X_train, 0.9)
    # Drop the features from both the training and testing set
    X_train = X_train.drop(columns=to_drop_9)
    X_test = X_test.drop(columns=to_drop_9)
    y_train = y_train  # Adjust as needed
    print("Using threshold = 0.9")
    print("Dropped features:", to_drop_9)
else:
    print("Invalid feature threshold. Please choose either 0.8 or 0.9.")

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=random_seed)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_tuned = RandomForestClassifier(
    n_estimators=136,        # Best hyperparameter for the number of trees
    max_depth=13,            # Best hyperparameter for the maximum depth of the trees
    max_features='sqrt',     # Use square root of features for splitting
    min_samples_split=2,     # Best hyperparameter for minimum samples required to split a node
    min_samples_leaf=2,      # Best hyperparameter for minimum samples required to be at a leaf node
    bootstrap=False,         # Whether bootstrap samples are used when building trees
    random_state=42          # Set the seed for reproducibility
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



# Create a new column for the misclassification count
misclassified_df = misclassified_df.groupby(['actual_label', 'predicted_label']).size().reset_index(name='count')

# Find the most frequent misclassifications
misclassified_sorted = misclassified_df.sort_values(by='count', ascending=False)

# Get the misclassification with the highest proportion
highest_misclassification = misclassified_sorted.iloc[0]
actual = highest_misclassification['actual_label']
predicted = highest_misclassification['predicted_label']
count = highest_misclassification['count']

# Calculate the percentage of the highest misclassification proportion
total_misclassified = misclassified_sorted['count'].sum()
percentage = (count / total_misclassified) * 100

# Print the misclassification with the highest proportion and its percentage
print(f"Misclassification with the highest proportion: Actual {actual} → Predicted {predicted}")
print(f"Percentage of misclassified instances: {percentage:.2f}%")

# Plotting the highest proportion of misclassifications
plt.figure(figsize=(10, 6))
plt.bar(misclassified_sorted['actual_label'].astype(str) + ' → ' + misclassified_sorted['predicted_label'].astype(str),
        misclassified_sorted['count'], color='skyblue')

# Set labels and title
plt.xlabel('Misclassified Label (Actual → Predicted)', fontsize=12)
plt.ylabel('Count of Misclassified Instances', fontsize=12)
plt.title('Highest Proportion of Misclassified Instances by Actual vs Predicted Labels', fontsize=14)

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
# plt.show()

# Identify correctly classified instances where both true label and predicted label are 3
correctly_classified_indices = np.where((y_test == 3) & (y_test_pred == 3))[0]  # Indices of correctly classified rows where y_test = 3 and y_test_pred = 3

correctly_classified_df = pd.DataFrame(X_test_scaled[correctly_classified_indices])

correctly_classified_row = correctly_classified_df.iloc[0].values.reshape(1, -1)

# Identify misclassified instances where true label is 3 and predicted label is 4
misclassified_indices = np.where((y_test != y_test_pred) & (y_test == 3) & (y_test_pred == 4))[0]  # Indices of misclassified rows where y_test = 3 and y_test_pred = 4

misclassified_df = pd.DataFrame(X_test_scaled[misclassified_indices])

# If you want to extract just one row, you can use `.iloc[0]` to select the first row:
misclassified_row = misclassified_df.iloc[0].values.reshape(1, -1)  # Get the first misclassified row where y_test = 3 and y_test_pred = 4
 

print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of random_row_values: {misclassified_row.shape}")


# Step 1: Initialize LIME explainer
explainer = LimeTabularExplainer(
    X_train_scaled,  # Training data to help with explaining the model (directly use the NumPy array)
    mode="classification",  # We are working with a classification problem
    feature_names=X_train_scaled.columns.tolist() if isinstance(X_train_scaled, pd.DataFrame) else ['Feature ' + str(i) for i in range(X_train_scaled.shape[1])],  # List of feature names
    class_names=np.unique(y_train).astype(str),  # Class names (e.g., 0, 1, 2, ..., 11)
    discretize_continuous=True,  # Discretize continuous features if necessary
    random_state=42  # Ensures reproducibility of the explanation
)

# Step 2: Generate explanation for the random misclassified row
explanation = explainer.explain_instance(
    misclassified_row[0],  # The actual row (as a 1D array inside a 2D structure)
    rf_tuned.predict_proba,  # The Random Forest model's predict_proba function to get probabilities
    num_features=7  # Show the top 7 important features
)
correct_feature_importances = {}
for feature, weight in explanation.as_list():
    correct_feature_importances[feature] = correct_feature_importances.get(feature, 0) + abs(weight)

# Step 2: Generate explanation for the correctly classified row
explanation = explainer.explain_instance(
    correctly_classified_row[0],  # The actual row (as a 1D array inside a 2D structure)
    rf_tuned.predict_proba,  # The Random Forest model's predict_proba function to get probabilities
    num_features=7  # Show the top 7 important features
)
# Generate explanations for misclassified class 3 -> class 4
misclassified_feature_importances = {}
for feature, weight in explanation.as_list():
    misclassified_feature_importances[feature] = misclassified_feature_importances.get(feature, 0) + abs(weight)



# Convert to sorted lists
correct_features, correct_weights = zip(*sorted(correct_feature_importances.items(), key=lambda x: x[1], reverse=True))
misclassified_features, misclassified_weights = zip(*sorted(misclassified_feature_importances.items(), key=lambda x: x[1], reverse=True))
#%%
# Plot


# Plot with renamed features
num_feat = 6
plt.figure(figsize=(12, 6))
plt.barh(correct_features[0:num_feat], correct_weights[0:num_feat], color='green', alpha=0.5, label='Correctly Classified Class 3')
plt.barh(misclassified_features[0:num_feat], misclassified_weights[0:num_feat], color='red', alpha=0.3, label='Misclassified as Class 4')
plt.xlabel('Feature Importance')
plt.title('Feature Importance Comparison: Class 3 Correct vs Misclassified as Class 4')
plt.legend()
# Adjust the left margin
plt.subplots_adjust(left=0.25)  # Increase the left margin (default is ~0.125)
plt.show()