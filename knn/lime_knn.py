# %%
import csv
import math
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from sklearn import neighbors



random_seed = 31
#%%
from imblearn.over_sampling import SMOTE
from collections import Counter
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

train.columns = train.columns.str.replace(' ', '') 
test.columns = test.columns.str.replace(' ', '') 

X_train = train.iloc[:, :-2] # remove id and label col
y_train = train[['label']] -1

X_test = test.iloc[:, :-2]
y_test = test[['label']]  -1
smote = SMOTE(random_state=random_seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, metric='manhattan')
knn_model = knn.fit(X_train_smote, y_train_smote) 


#%%
# For LIME to work

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
features = X_train.columns.tolist()
#%%
# on tuned oversampled model
# For single instance/index
# idx = 17

# explainer = LimeTabularExplainer(
#     X_train_np,  # Use the NumPy array
#     mode="classification",
#     feature_names=features,
#     class_names=np.unique(y_train).astype(str),
#     discretize_continuous=True
# )

# # Convert the test instance to a 1D numpy array (this ensures it's in the correct format for LIME)
# test_instance = X_test_np[idx] 

# # Generate an explanation for the test instance
# explanation = explainer.explain_instance(
#     test_instance,
#     knn_model.predict_proba,  # Use predict_proba to get class probabilities
#     num_features=10,
#     top_labels=2
# )


# # Visualize the explanation
# explanation.show_in_notebook()

#%%
# Need run this

# Define the wrapper to add predict_proba functionality
class ProbabilisticWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Ensure the model supports predict_proba
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support predict_proba.")



# Wrap the model
knn_wrapper = ProbabilisticWrapper(knn_model)

# Initialize LIME explainer
explainer = LimeTabularExplainer(
    X_train_np,
    mode="classification",
    feature_names=features,
    class_names=[f"Class {i}" for i in range(12)],
    discretize_continuous=True
)
#%%

# LIME explanations for Class Label 3
misclassified_df = pd.read_csv("./knn_misclassifications.csv")

correct_classifications = pd.read_csv("./knn_correct_classifications.csv")
correct_classifications['True Label'] = correct_classifications['True Label'] - 1
correct_classifications['Predicted Label'] = correct_classifications['Predicted Label'] - 1


# Filter correctly classified class 3
correct_class3_indices = correct_classifications[(correct_classifications['True Label'] == 3) & (correct_classifications['Predicted Label'] == 3)]['Index'].values

# Filter misclassified: True class 3, Predicted class 4
misclassified_class3_as4_indices = misclassified_df[
    (misclassified_df['True Label'] == 3) & (misclassified_df['Predicted Label'] == 4)
]['Index'].values

# Assuming `feature_names` is a list of actual feature names corresponding to the columns in X_train
feature_names = X_train.columns.tolist()  # Adjust as needed for your data source

# Generate explanations for correctly classified class 3
correct_feature_importances = {}
for idx in correct_class3_indices:
    test_instance = X_test_np[idx]
    explanation = explainer.explain_instance(test_instance, knn_wrapper.predict_proba, num_features=7)
    for feature, weight in explanation.as_list():
        # Use signed weights instead of absolute weights
        correct_feature_importances[feature] = correct_feature_importances.get(feature, 0) + weight

# Generate explanations for misclassified class 3 -> class 4
misclassified_feature_importances = {}
for idx in misclassified_class3_as4_indices:
    test_instance = X_test_np[idx]
    explanation = explainer.explain_instance(test_instance, knn_wrapper.predict_proba, num_features=7)
    for feature, weight in explanation.as_list():
        # Use signed weights instead of absolute weights
        misclassified_feature_importances[feature] = misclassified_feature_importances.get(feature, 0) + weight
#%%

# Visualization

# Convert to sorted lists
correct_features, correct_weights = zip(*sorted(correct_feature_importances.items(), key=lambda x: abs(x[1]), reverse=True))
misclassified_features, misclassified_weights = zip(*sorted(misclassified_feature_importances.items(), key=lambda x: abs(x[1]), reverse=True))

# Plot comparison
num_feat = 6  # Adjust as needed
plt.figure(figsize=(12, 6))
plt.barh(correct_features[:num_feat], correct_weights[:num_feat], color='green', alpha=0.5, label='Correctly Classified Class 3')
plt.barh(misclassified_features[:num_feat], misclassified_weights[:num_feat], color='red', alpha=0.3, label='Misclassified as Class 4')
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Add a line at 0 for reference
plt.xlabel('Feature Weight')
plt.title('Feature Importance Comparison (Actual Label: 3): Correctly Classified as Class 3 vs Misclassified as Class 4')
plt.legend()
plt.show()


#%%

# for correct classifications

# Load the misclassifications CSV
misclassified_df = pd.read_csv("./knn_misclassifications.csv")

# Create a set of misclassified indices for quick lookup
misclassified_indices = set(misclassified_df["Index"])

# Correctly classified indices are those not in the misclassified set
correct_indices = [i for i in range(len(y_test)) if i not in misclassified_indices]

# Create a DataFrame for the correctly classified instances
correct_classifications = pd.DataFrame({
    "Index": correct_indices,
    "True Label": [y_test.iloc[i] for i in correct_indices],
    "Predicted Label": [knn_model.predict(X_test)[i] for i in correct_indices],
})

# Simulated DataFrame
correct_classifications = pd.read_csv("./knn_correct_classifications.csv")

# Extract numeric part from "True Label"
correct_classifications["True Label"] = correct_classifications["True Label"].str.extract(r'(\d+)').astype(int)
# correct_classifications.to_csv("./knn_correct_classifications.csv", index=False)

print("Correct classifications saved to 'knn_correct_classifications.csv'")
