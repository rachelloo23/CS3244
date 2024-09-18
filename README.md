# CS3244 - Machine Learning Model Comparison

This repository contains implementations of various machine learning models for the CS3244 project. The objective is to compare traditional machine learning algorithms with deep learning models, evaluating their performance on a shared dataset.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Models](#models)
- [Data Preprocessing](#data-preprocessing)
- [Configuration](#configuration)
- [Results](#results)

## Directory Structure

├── data/ │ ├── raw/ # Raw dataset files (e.g., CSVs, images) │ ├── processed/ # Preprocessed datasets used for training ├── models/ │ ├── knn/ # k-Nearest Neighbors implementation │ │ ├── knn.py │ │ └── results/ # Results and logs for kNN │ ├── decision_tree/ # Decision Tree implementation │ │ ├── decision_tree.py │ │ └── results/ # Results and logs for Decision Tree │ ├── random_forest/ # Random Forest implementation │ │ ├── random_forest.py │ │ └── results/ # Results and logs for Random Forest │ ├── gradient_boosting/ # Gradient Boosting implementation │ │ ├── gradient_boosting.py │ │ └── results/ # Results and logs for Gradient Boosting │ ├── deep_learning/ │ │ ├── cnn/ # Convolutional Neural Network (CNN) implementation │ │ │ ├── cnn.py │ │ │ └── results/ # Results and logs for CNN │ │ ├── lstm/ # Long Short-Term Memory (LSTM) implementation │ │ │ ├── lstm.py │ │ │ └── results/ # Results and logs for LSTM │ │ ├── autoencoder/ # Autoencoder implementation │ │ │ ├── autoencoder.py │ │ │ └── results/ # Results and logs for Autoencoder ├── notebooks/ # Jupyter notebooks for experimentation ├── scripts/ # Utility scripts for preprocessing, evaluation │ ├── preprocess.py # Script for data preprocessing │ ├── evaluation.py # Script for evaluation metrics ├── README.md # This file ├── requirements.txt # List of required packages ├── config.yaml # Configuration file for hyperparameters └── results/ ├── summary.csv # Summary of model performance metrics