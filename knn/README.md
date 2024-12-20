# K-Nearest Neighbors (KNN) Model

This repository contains the implementation and evaluation of a k-Nearest Neighbors (KNN) classification model. The focus is on explaining model predictions using Local Interpretable Model-agnostic Explanations (LIME) and visualizing the results. Below is a detailed guide on how to reproduce the experiments.

## Files

### Python Scripts
- **`knn.py`**: Contains the implementation of the KNN model, including training and evaluation.
- **`final_knn_model.py`**: The finalized version of the KNN model, optimized and used for generating results.
- **`lime_knn.py`**: Script to generate LIME explanations for correctly classified and misclassified instances.

### Data Files
- **`knn_correct_classifications.csv`**: CSV file containing indices and labels of correctly classified instances.
- **`knn_misclassifications.csv`**: CSV file containing indices and labels of misclassified instances.

### Visualizations
- **`lime plot.png`**: A bar chart comparing feature importances for correctly classified and misclassified instances.

### Configuration
- **`requirements.txt`**: List of required Python libraries to set up the environment.

## Reproducing Results

### Step 1: Train and Evaluate the Model
Run `knn.py` to train the KNN model and evaluate its performance:
```bash
python knn.py
```

### Step 2: Generate LIME Explanations
Run `lime_knn.py` to generate explanations for correctly classified and misclassified instances:
```bash
python lime_knn.py
```

### Step 3: Visualize Feature Importances
The script `lime_knn.py` generates a bar chart (`lime plot.png`) comparing the feature importances for correctly classified and misclassified instances. Open the file to view the visualization.

## Notes
- The scripts assume that the data has been preprocessed and split into training and test sets. Ensure these are correctly defined in the code.
- Modify paths and parameters in the scripts as needed to match your environment.
