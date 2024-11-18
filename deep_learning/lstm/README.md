# LSTM Model for Activity Recognition

This repository contains the implementation of an LSTM-based neural network model for activity recognition using time-series data. The project includes data preprocessing, model architecture definition, hyperparameter tuning, model training, and evaluation.

## Folder Structure
```
.
├── config/
│   └── config.yaml             # Contains best model hyperparameters
├── fig/                        # Figures and plots from training and evaluation
│   └── *.png                   # PNG images (e.g., loss curves, accuracy plots)
├── log/                        # Logs from the hyperparameter tuning process (not uploaded)
├── model/
│   └── lstm_model.h5           # Saved trained model
├── results/
│   ├── testResults.txt         # Evaluation results on the test dataset
│   └── hparams_table.csv       # Hyperparameters and corresponding performance metrics
├── lstm.py                     # Contains the LSTM model architecture definition
├── main.py                     # Script for hyperparameter tuning process
├── raw_data_preprocess.py      # Data loading and preprocessing (duplicate from scripts folder)
├── test.py                     # Script for training and testing the model
```

### Key Components

- **Data Preprocessing (`raw_data_preprocess.py`):**
  - Loads raw data from sensor readings.
  - Cleans and normalizes data.
  - Prepares data for training by creating appropriate data structures.

- **Model Architecture (`lstm.py`):**
  - Defines the LSTM model architecture, including:
    - Masking layers to handle variable-length sequences.
    - Layer normalization for input stabilization.
    - LSTM layers to capture temporal dependencies.
    - Dense layers with ReLU activation functions.
    - Dropout layers for regularization.
    - Output layer with softmax activation for classification.

- **Hyperparameter Tuning (`main.py`):**
  - Utilizes Ray Tune and Optuna to perform hyperparameter optimization.
  - Explores different combinations of hyperparameters such as:
    - Number of LSTM units.
    - Dropout rate.
    - Batch size.
    - Number of epochs.
  - Stores the best hyperparameters in `config/config.yaml`.

- **Model Training and Evaluation (`test.py`):**
  - Trains the LSTM model using the best hyperparameters.
  - Evaluates the model on a separate test dataset.
  - Generates evaluation metrics including confusion matrices and classification reports.
  - Saves evaluation results in `results/testResults.txt`.

- **Visualization (`fig/`):**
  - Stores plots and figures generated during training and evaluation.
  - Visualizations include training loss curves, accuracy over epochs, and more.