import os
import math
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.metrics import AUC, F1Score

from raw_data_preprocess import DataCatalogProcessor, DataLoader
from lstm import LSTMModel  
from main import *

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data'))
RAW_DATA_PATH = os.path.join(DATA_PATH, 'RawData')
CONFIG_PATH = "./config/"


def load_config(config_path, config_name):
    """
    Load configuration from a YAML file.

    Parameters:
    - config_path: Path to the directory containing the config file.
    - config_name: Name of the config file.

    Returns:
    - config: A dictionary containing the configuration parameters.
    """
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def create_tf_dataset(
    data_catalog, path, indices, padded_length, num_classes, batch_size, repeat=True
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from the filtered data (data_catalog with the corresponding indices).

    Parameters:
    - data_catalog: DataFrame with metadata to load the data.
    - path: Path to the raw data files.
    - indices: The indices of the data to load.
    - padded_length: The length to pad the time steps to.
    - num_classes: Number of classes for one-hot encoding.
    - batch_size: The batch size for the dataset.
    - repeat: Whether to repeat the dataset indefinitely.

    Returns:
    - padded_data: The prepared TensorFlow dataset.
    """
    dataLoader = DataLoader(data_catalog, path)
    data = dataLoader.load_experiment_data(indices)

    if repeat:
        data = data.shuffle(buffer_size=len(indices))

    # One-hot encode the labels
    data = data.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))

    # Pad and batch the dataset
    padded_data = data.padded_batch(
        batch_size=batch_size,
        padded_shapes=([padded_length, 6], [num_classes])
    )

    if repeat:
        return padded_data.repeat()
    else:
        return padded_data


def eval_lstm(model, test_dataset, number_of_eval_sample, batch_size, num_classes):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model: The trained model to evaluate.
    - test_dataset: The test dataset to evaluate the model on.
    - number_of_eval_sample: Total number of samples in the test dataset.
    - batch_size: The batch size used during evaluation.
    - num_classes: Number of classes.

    Returns:
    - eval_metrics: Dictionary containing evaluation metrics.
    """
    steps = math.ceil(number_of_eval_sample / batch_size)

    print('#######################')
    print('Evaluating the model...')
    print('#######################')

    # Evaluate the model and get metrics
    eval_metrics = model.evaluate(test_dataset, steps=steps, return_dict=True)

    print('#######################')
    print('Evaluation completed...')
    print('#######################')
    print("Test Loss:", eval_metrics['loss'])
    print("Test Accuracy:", eval_metrics['accuracy'])
    print("Test AUC:", eval_metrics['auc'])
    print("Test F1 Micro (from metrics):", eval_metrics['f1_micro'])
    print("Test F1 Macro (from metrics):", eval_metrics['f1_macro'])
    print("Test F1 Weighted (from metrics):", eval_metrics['f1_weighted'])

    # Collect true labels and predictions
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_dataset.take(steps):
        # Get model predictions
        y_pred_batch = model.predict(x_batch)
        # Convert predictions and labels to class indices
        y_true_batch = tf.argmax(y_batch, axis=1).numpy()
        y_pred_batch = tf.argmax(y_pred_batch, axis=1).numpy()

        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Compute F1 Scores using sklearn
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print("F1 Score (Micro):", f1_micro)
    print("F1 Score (Macro):", f1_macro)
    print("F1 Score (Weighted):", f1_weighted)

    return eval_metrics


def train_lstm(model, train_dataset, number_of_training, batch_size, epochs=10):
    """
    Train the LSTM model on the training dataset.

    Parameters:
    - model: The LSTM model to train.
    - train_dataset: The training dataset.
    - number_of_training: Total number of training samples.
    - batch_size: The batch size for training.
    - epochs: Number of epochs to train.

    Returns:
    - history: The history object containing training metrics.
    """
    steps_per_epoch = number_of_training // batch_size

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )

    return history


def plot_training_history(history):
    """
    Plot training loss and metrics over epochs.

    Parameters:
    - history: The history object returned from model training.
    """
    # Plot training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(alpha=0.35)
    plt.legend()
    plt.savefig('./fig/trainLoss.png', dpi=400)
    plt.show()

    # Plot training accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(alpha=0.35)
    plt.legend()
    plt.savefig('./fig/trainAccuracy.png', dpi=400)
    plt.show()

    # Plot F1 Macro
    if 'f1_macro' in history.history:
        plt.figure()
        plt.plot(history.history['f1_macro'], label='F1 Macro')
        plt.title('Training F1 Macro')
        plt.ylabel('F1 Macro')
        plt.xlabel('Epoch')
        plt.grid(alpha=0.35)
        plt.legend()
        plt.savefig('./fig/trainF1_Macro.png', dpi=400)
        plt.show()

    # Plot F1 Weighted
    if 'f1_weighted' in history.history:
        plt.figure()
        plt.plot(history.history['f1_weighted'], label='F1 Weighted')
        plt.title('Training F1 Weighted')
        plt.ylabel('F1 Weighted')
        plt.xlabel('Epoch')
        plt.grid(alpha=0.35)
        plt.legend()
        plt.savefig('./fig/trainF1_Weighted.png', dpi=400)
        plt.show()


def test():
    """
    Main function to train and evaluate the LSTM model.
    """
    # Load configuration parameters
    config = load_config(CONFIG_PATH, "config.yaml")
    epochs = config['epochs']
    lstm_units = config['lstm_units']
    dropout_rate = config['dropout_rate']
    batch_size = config['batch_size']

    # Prepare data catalog
    data_catalog_processor = DataCatalogProcessor(DATA_PATH, RAW_DATA_PATH)
    data_catalog = prepare_data_catalog(data_catalog_processor)
    data_catalog = data_catalog.reset_index(drop=True)

    # Determine maximum padding length and number of classes
    max_padding = data_catalog['window_size'].max()
    num_classes = data_catalog['activity_id'].nunique()

    # Encode labels
    label_encoder = LabelEncoder()
    data_catalog['encoded_labels'] = label_encoder.fit_transform(data_catalog['activity_id'])

    # Split data into training and test sets
    train_data_catalog, test_data_catalog = train_test_split(
        data_catalog,
        test_size=0.2,
        random_state=31,
        stratify=data_catalog['encoded_labels']
    )

    train_idx = train_data_catalog.index
    test_idx = test_data_catalog.index

    # Create TensorFlow datasets for training and testing
    train_dataset = create_tf_dataset(
        data_catalog,
        RAW_DATA_PATH,
        train_idx,
        max_padding,
        num_classes,
        batch_size,
        repeat=True
    )
    test_dataset = create_tf_dataset(
        data_catalog,
        RAW_DATA_PATH,
        test_idx,
        max_padding,
        num_classes,
        batch_size,
        repeat=False
    )

    # Build and train the model
    model = lstm_builder(num_classes, lstm_units, dropout_rate)
    history = train_lstm(model, train_dataset, len(train_idx), batch_size, epochs)

    # Save the trained model
    model.save('./model/lstm_model.h5')

    # Plot training history
    plot_training_history(history)

    # Evaluate the model on the test dataset
    eval_metrics = eval_lstm(model, test_dataset, len(test_idx), batch_size, num_classes)


if __name__ == '__main__':
    test()