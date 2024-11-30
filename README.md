# CS3244 - Machine Learning Project

This repository explores two distinct approaches to human activity recognition using the [**UCI Human Activity Recognition (HAR)** dataset](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions). The project investigates the performance trade-offs between traditional machine learning methods leveraging expert-processed features and deep learning models applied directly to raw time-series data.

## Objectives

1. **Traditional Machine Learning with Expert Features**:  
   This approach uses the preprocessed UCI HAR dataset, enriched with domain-specific feature engineering by experts. Models such as k-Nearest Neighbors (kNN), Decision Trees, and Random Forests are applied to evaluate their effectiveness in leveraging structured data.

2. **Deep Learning on Raw Time-Series Data**:  
   A deep learning approach, particularly using Long Short-Term Memory (LSTM) networks, processes raw time-series data without requiring manual feature engineering. This end-to-end learning paradigm allows the model to autonomously extract features and learn temporal dependencies.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- \
## Directory Structure
```
.
├── data/
│   ├── RawData/                                # Raw data from UCI HAR dataset
│   ├── Test/                                   # Testing dataset
│   ├── Train/                                  # Training dataset
│   ├── processed/                              # Processed datasets with selected features
│   │   ├── train_8.csv                         # Train set (threshold > 0.8)
│   │   ├── train_9.csv                         # Train set (threshold > 0.9)
│   │   ├── test_8.csv                          # Test set (threshold > 0.8)
│   │   ├── test_9.csv                          # Test set (threshold > 0.9)
│   ├── activity_labels.txt                     # Activity label descriptions
│   ├── features.txt                            # Feature names
│   └── features_info.txt                       # Description of features
├── feature_engineering/
│   ├── main.py                                 # Main script for feature engineering
│   ├── high_corr_features_8.csv                # Highly correlated features (>0.8)
│   ├── high_corr_features_9.csv                # Highly correlated features (>0.9)
├── models/
│   ├── knn/                                    # k-Nearest Neighbors implementation
│   │   ├── knn.py                              # Main kNN implementation
│   │   └── results/                            # Results and logs for kNN
│   ├── decision_tree/                          # Decision Tree implementation
│   │   ├── decision_tree.py                    # Main Decision Tree implementation
│   │   └── results/                            # Results and logs for Decision Tree
│   ├── random_forest/                          # Random Forest implementation
│   │   ├── random_forest.py                    # Main Random Forest implementation
│   │   └── results/                            # Results and logs for Random Forest
│   ├── gradient_boosting/                      # Gradient Boosting implementation
│   │   ├── config/                             # Configuration files for the model
│   │   │   ├── config.yaml                     # Optimal hyperparameters
│   │   ├── main.py                             # Main Gradient Boosting script
│   │   ├── tune.py                             # Hyperparameter tuning script
│   │   └── results/                            # Results and logs for Gradient Boosting
│   │       ├── xgb_tune_results.csv            # Results from pre-standardization tuning
│   │       └── xgb_tune_results_2.csv          # Results from post-standardization tuning
│   ├── deep_learning/
│   │   └── lstm/                               # Long Short-Term Memory (LSTM) implementation
│   │       ├── lstm.py                         # Main LSTM implementation
│   │       └── results/                        # Results and logs for LSTM
├── scripts/                                    # Utility scripts
│   ├── cs3244_eda.py                           # Script for exploratory data analysis (EDA)
│   └── preprocess.py                           # Script for data preprocessing
├── README.md                                   # Project description and usage guide
├── .gitignore                                  # Files and directories to be ignored by Git
└── requirements.txt                            # Python dependencies
```
## Prerequisites
Before setting up the project, make sure you have the following installed:
- [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
## Setup Instructions
### 1. Clone the Repository
Clone the project from GitHub to your local machine:
```
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```
### 2. Set Up a Virtual Environment Using Conda
Create and activate a virtual environment:
```
conda create -n CS3244 python=3.11
conda activate CS3244
```
### 3. Install Dependencies
Key packages included in the ```requirements.txt```:
```
conda install --file requirements.txt
```
#### PackageNotFoundError
If you encounter this error and it said certain packages are not available from the current channel, 
run the following command:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
```
then try step 3 again.

### 4. Environment Setup Complete
To start working in the Conda environment, use the following command:
```
conda activate CS3244
```


