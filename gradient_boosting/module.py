import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer
import yaml

def classAcc(confusion_matrix):
    """
    Calculate per-class accuracy and misclassification rates from a confusion matrix.

    Args:
        confusion_matrix (numpy.ndarray): A square matrix where the element at [i, j]
                                           represents the count of samples with true label i
                                           predicted as label j.

    Returns:
        tuple: Two dictionaries:
            - per_class_accuracies: Accuracy for each class (keys: class indices).
            - per_class_misses: Misclassification rate for each class (1 - accuracy).
    """
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    total_samples = np.sum(confusion_matrix)
    per_class_accuracies = {}
    per_class_misses = {}

    for idx in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[idx, idx]
        fn = np.sum(confusion_matrix[idx, :]) - tp
        fp = np.sum(confusion_matrix[:, idx]) - tp
        tn = total_samples - (tp + fn + fp)

        accuracy = (tp + tn) / total_samples
        per_class_accuracies[idx] = accuracy
        per_class_misses[idx] = 1 - accuracy

    return per_class_accuracies, per_class_misses

def writeResults(model, X_train, y_train, X_test, y_test, filename):
    """
    Evaluate the model and save the confusion matrix, classification reports, and per-class metrics to a text file.

    Parameters:
    - model: Trained model to evaluate.
    - X_train: Training feature set.
    - y_train: Training labels.
    - X_test: Test feature set.
    - y_test: Test labels.
    - filename: Name of the output text file (string).
    """
    # Evaluate and print training and test set scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('Model - Training set score: {:.4f}'.format(train_score))
    print('Model - Test set score: {:.4f}'.format(test_score))

    # Predict on the training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute confusion matrix for the test set
    cm = confusion_matrix(y_test, y_test_pred)

    # Compute per-class metrics using classAcc
    per_class_accuracies, per_class_misses = classAcc(cm)

    # Generate classification reports
    report_train = classification_report(y_train, y_train_pred)
    report_test = classification_report(y_test, y_test_pred)

    # Save confusion matrix, classification reports, and per-class metrics to a text file
    with open(filename, 'w') as f:
        # Write confusion matrix
        f.write('Confusion Matrix for Test Set:\n')
        f.write(str(cm))
        f.write('\n\n')

        # Write classification report for the training set
        f.write('Classification Report for Training Set:\n')
        f.write(report_train)
        f.write('\n\n')

        # Write classification report for the test set
        f.write('Classification Report for Test Set:\n')
        f.write(report_test)
        f.write('\n\n')

        # Write per-class metrics
        f.write('Per-Class Metrics:\n')
        f.write('Class Accuracies:\n')
        for cls, acc in per_class_accuracies.items():
            f.write(f'Class {cls}: Accuracy = {acc:.4f}\n')
        f.write('\nClass Misclassification Rate:\n')
        for cls, miss in per_class_misses.items():
            f.write(f'Class {cls}: Misclassification Rate = {miss:.4f}\n')

    # Indicate that the results have been saved
    print(f'Results have been saved to {filename}')

def load_config(config_path, config_name):
    """
    Load a configuration file in YAML format.

    Args:
        config_path (str): The directory path where the configuration file is located.
        config_name (str): The name of the configuration file (including extension, e.g., 'config.yaml').

    Returns:
        dict: The loaded configuration as a Python dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    # Construct the full path to the configuration file
    config_file = os.path.join(config_path, config_name)

    # Open and read the YAML configuration file
    with open(config_file, 'r') as file:
        # Load the YAML content safely into a dictionary
        config = yaml.safe_load(file)

    # Return the loaded configuration
    return config

def misclass_analysis(y_test, X_test, target_class, model):
    """
    Analyze misclassifications for a specific class in a test dataset.

    Args:
        y_test (pd.Series): True labels for the test dataset.
        X_test (np.ndarray): Feature set for the test dataset.
        target_class (int): The class for which misclassifications are analyzed.
        model: A trained model with a `predict` method.

    Returns:
        dict: A dictionary where keys are misclassified classes and values are lists of indices
              of samples misclassified as those classes.
    """
    # Get indices of samples belonging to the target class
    target_class_indices = y_test[y_test == target_class].index.to_list()

    # Dictionary to store misclassified samples
    misclassified_results = {}

    # Iterate through samples of the target class
    for idx in target_class_indices:
        # Predict the class for the current sample
        pred = model.predict(X_test[idx].reshape(1, -1))[0]
        
        # If misclassified, record the result
        if pred != target_class:
            if pred not in misclassified_results:
                misclassified_results[pred] = []
            misclassified_results[pred].append(idx)
    
    return misclassified_results