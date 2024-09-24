# %%
# import lib
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# %%


# %% [markdown]
# sections:
# 1. data import/preparation
# 2. data exploration
# 3. feature analysis
# 4. visualisations
# 5. dimensionality reduction

# %% [markdown]
# # data import and preparation

# %%
cur_dir = os.getcwd()

# unzip dataset
with zipfile.ZipFile('smartphone+based+recognition+of+human+activities+and+postural+transitions.zip', 'r') as zip_ref:
    zip_ref.extractall(cur_dir)

    

# %%
# load training data
X_train = pd.read_csv(os.path.join(cur_dir, 'Train', 'X_train.txt'), delim_whitespace=True, header=None)
y_train = pd.read_csv(os.path.join(cur_dir, 'Train', 'y_train.txt'), delim_whitespace=True, header=None)
subject_train = pd.read_csv(os.path.join(cur_dir, 'Train', 'subject_id_train.txt'), delim_whitespace=True, header=None)

# load test data
X_test = pd.read_csv(os.path.join(cur_dir, 'Test', 'X_test.txt'), delim_whitespace=True, header=None)
y_test = pd.read_csv(os.path.join(cur_dir, 'Test', 'y_test.txt'), delim_whitespace=True, header=None)
subject_test = pd.read_csv(os.path.join(cur_dir, 'Test', 'subject_id_test.txt'), delim_whitespace=True, header=None)

# load feature names
features = pd.read_csv(os.path.join(cur_dir, 'features.txt'), delim_whitespace = True, header=None)

# %%
combined_X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
combined_y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
combined_subj = pd.concat([subject_test, subject_train], axis=0).reset_index(drop=True)



# %%
combined_X.columns = features[0]

# %%
df_raw = pd.concat([combined_subj, combined_y, combined_X], axis=1)


# %%
df_raw.columns = ['subject_id', 'activity'] + list(features[0])

# %%
# check if missing values or duplicates
missing = df_raw.isnull().sum()
print(missing[missing>0])

# %%
duplicate = df_raw[df_raw.duplicated()]
print(duplicate.shape[0])

# %% [markdown]
# # data exploration

# %%
df = df_raw.copy()

# %%
print("df.shape:", df.shape)
display(df.head())

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x='activity', data=df)
plt.title('Distribution of Activity Labels')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# # feature selection

# %%
X = df.drop(['subject_id', 'activity'], axis=1)
y = df['activity']

# %%
# anova f test for feature importance
f_values, p_values = f_classif(X, y)

# %%
feature_scores = pd.DataFrame({'Feature': X.columns, 'F-Value': f_values, 'p-Value': p_values})
feature_scores = feature_scores.sort_values(by='F-Value', ascending=False)

# %%
# Select top features based on p-value threshold 
threshold = 0.05/len(feature_scores)
significant_features = feature_scores[feature_scores['p-Value'] < threshold]
print(f"Number of significant features: {len(significant_features)}")
# all 561 features are statistically significant 


# %%
# Plot Top 10 Features by F-Value
top_features = feature_scores.head(10)
plt.figure(figsize=(14, 6))
sns.barplot(x='F-Value', y='Feature', data=top_features)
plt.title('Top 10 Features by ANOVA F-Value')
plt.xlabel('F-Value')
plt.ylabel('Feature')
plt.show()

# %% [markdown]
# # visualisations

# %%
# Get the top 10 features by F-Value
top_features = feature_scores.sort_values(by='F-Value', ascending=False).head(10)
selected_features = top_features['Feature'].tolist()

print(f"Top 10 Selected Features by F-Value:\n{selected_features}")


# %% [markdown]
# ## visualisations for selected features by f-value

# %%
# Filter out only the one-dimensional features
one_dim_features = [feature for feature in selected_features if df[feature].ndim == 1]

print(f"One-Dimensional Selected Features: {one_dim_features}")


# %%
# Histograms for one-dimensional selected features
for feature in one_dim_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=feature, kde=True, hue='activity', palette='Set2', element='step')
    plt.title(f'Distribution of {feature} by Activity')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# %%
# Correlation Heatmap for Top 10 Features
corr_matrix = df[selected_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Top 10 Selected Features')
plt.show()

# %%

# PCA using Top 10 Features
X_selected = df[selected_features]
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_selected)

df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
df_pca['activity'] = df['activity']

# Visualize PCA results with a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='activity', data=df_pca, palette='Set2', s=60)
plt.title('PCA of Activity Data (Top 10 Features)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Activity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
# t-SNE using Top 10 Features
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X_selected)

# Create a DataFrame with t-SNE results and activity labels
df_tsne = pd.DataFrame(data=tsne_result, columns=['t-SNE1', 't-SNE2'])
df_tsne['activity'] = df['activity']

# Visualize t-SNE results with a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='activity', data=df_tsne, palette='Set2', s=60)
plt.title('t-SNE of Activity Data (Top 10 Features)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Activity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %% [markdown]
# # summary

# %%


# %%


# %%


# %%



