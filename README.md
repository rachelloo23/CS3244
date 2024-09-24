# CS3244 - Machine Learning Model Comparison
This repository contains implementations of various machine learning models for the CS3244 project. The objective is to compare traditional machine learning algorithms with deep learning models, evaluating their performance on a shared dataset.
## Table of Contents

- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)]
- [Models](#models)
- [Data Preprocessing](#data-preprocessing)
- [Configuration](#configuration)
- [Results](#results)
## Directory Structure
```
.
├── data/
│   ├── activity_labels.txt
│   ├── features.txt
|   └── features_info.txt
├── models/
│   ├── knn/                  # k-Nearest Neighbors implementation
│   │   ├── knn.py
│   │   └── results/          # Results and logs for kNN
│   ├── decision_tree/        # Decision Tree implementation
│   │   ├── decision_tree.py
│   │   └── results/          # Results and logs for Decision Tree
│   ├── random_forest/        # Random Forest implementation
│   │   ├── random_forest.py
│   │   └── results/          # Results and logs for Random Forest
│   ├── gradient_boosting/    # Gradient Boosting implementation
│   │   ├── gradient_boosting.py
│   │   └── results/          # Results and logs for Gradient Boosting
│   ├── deep_learning/
│   │   ├── cnn/              # Convolutional Neural Network (CNN) implementation
│   │   │   ├── cnn.py
│   │   │   └── results/      # Results and logs for CNN
│   │   ├── lstm/             # Long Short-Term Memory (LSTM) implementation
│   │   │   ├── lstm.py
│   │   │   └── results/      # Results and logs for LSTM
│   │   ├── autoencoder/      # Autoencoder implementation
│   │   │   ├── autoencoder.py
│   │   │   └── results/      # Results and logs for Autoencoder
├── notebooks/                # Jupyter notebooks for experimentation
├── scripts/                  # Utility scripts for preprocessing, evaluation
│   ├── preprocess.py         # Script for data preprocessing
│   ├── evaluation.py         # Script for evaluation metrics
├── README.md                 # This file
├── .gitignore
├── requirements.txt          
└── config.yaml               # Configuration file for hyperparameters
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
When you're done and want to exit the environment, use:
```
conda deactivate
```
### 5. (Optional) Interactive Mode with JupyterLab
If you want to use an interactive environment, you can launch JupyterLab by running the following command:
```
jupyter lab
```
After executing this command, you should see output similar to the following in your terminal:
```
    To access the server, open this file in a browser:
        <YOUR_COMPUTER>/jupyter/runtime/<JUPYTER_SERVER>.html
    Or copy and paste one of these URLs:
        http://localhost:<PORT>/lab?token=<TOKEN_SEQ>
        http://127.0.0.1:<PORT>/lab?token=<TOKEN_SEQ>
```
To connect to JupyterLab, open a browser and copy the URL that looks like this:
```
http://localhost:<PORT>/lab?token=<TOKEN_SEQ>
```
Paste it into your browser’s address bar to access the JupyterLab interface.
