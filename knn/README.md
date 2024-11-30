# K-Nearest Neighbors (KNN) Project

This repository contains a project demonstrating the implementation and evaluation of a KNN model for classification. The project includes analysis of correct and misclassified data points and feature importance explanations using LIME.

## Files in This Repository
- `knn.py`: Implements the KNN model with hyperparameter tuning.
- `final_knn_model.py`: Contains the finalized KNN model.
- `lime_knn.py`: Generates LIME explanations for the model's predictions.
- `knn_correct_classifications.csv`: Dataset of correctly classified samples.
- `knn_misclassifications.csv`: Dataset of misclassified samples.
- `lime_plot.png`: Visualization of feature importances.

## Requirements
- Python 3.9.2
- Required libraries:
  ```
  numpy
  pandas
  scikit-learn
  matplotlib
  lime
  ```
  Install dependencies via `pip install -r requirements.txt`.

## Data
- `knn_correct_classifications.csv`: Contains indices and labels of samples classified correctly by the model.
- `knn_misclassifications.csv`: Contains indices and labels of samples misclassified by the model.

Ensure these files are located in the project root directory before running the scripts.

## How to Reproduce
1. **Clone the Repository:**
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up Environment:**
   Create a virtual environment and install dependencies:
   ```
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```

3. **Run KNN Model Training:**
   Execute the KNN training script:
   ```
   python knn.py
   ```

4. **Analyze Feature Importance with LIME:**
   Generate LIME explanations for correct and misclassified samples:
   ```
   python lime_knn.py
   ```
   The feature importance visualization will be saved as `lime_plot.png`.

## Outputs
- **Model Evaluation:**
  The script outputs classification metrics, including precision, recall, and F1 scores.
- **Feature Importance:**
  LIME-based visualizations provide insights into which features influenced the model's predictions.

## Notes
- Ensure `X_train`, `X_test`, and other data variables are correctly preprocessed before running the scripts.
- Modify file paths in the scripts if the project directory structure changes.

## Acknowledgments
This project uses the LIME library for interpretability and scikit-learn for model implementation.
