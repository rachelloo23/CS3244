import os
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split


sys.path.append(os.path.abspath('../../scripts'))

from raw_data_preprocess import DataCatalogProcessor, DataLoader
from lstm import LSTMModel
from sklearn.model_selection import KFold

# Define paths to data
DATA_PATH = '../../data'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'RawData')

def prepare_data_catalog(catalog: DataCatalogProcessor) -> pd.DataFrame:
    """
    Load the labels and activity data, merge them, and filter unwanted activities.

    Parameters:
    - catalog: DataCatalogProcessor object specifying the data to be loaded.
    """
    try:
        labels = catalog.load_labels()
        activity_labels = catalog.load_activity_labels()
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        exit(1)

    # # Compute window sizes and merge with activity names
    labels = catalog.compute_window_size(labels) # window_size = end_time - start_time 
    labels.sort_values(by='window_size', ascending=True, inplace=True)
    labels_with_activity = catalog.merge_labels_with_activity(labels, activity_labels)

    # Filter unwanted activities (e.g., those containing 'TO')
    unwanted_activities = activity_labels[activity_labels['activity_name'].str.contains('TO')]['activity_name'].tolist()
    filtered_data = catalog.drop_activity(labels_with_activity, unwanted_activities)

    return filtered_data

def create_tf_dataset(data_catalog, path, indices, padded_length) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from the filtered data (data_catalog with the corresponding index).

    Parameters:
    - data_catalog: DataFrame with metadata to load the data.
    - indices: The indices of the data to load.
    - batch_size: The batch size for the dataset.
    - padded_length: The length to pad the time steps to.
    """
    dataLoader = DataLoader(data_catalog, path)
    data = dataLoader.load_experiment_data(indices)

    # Pad and batch the dataset
    padded_data = data.padded_batch(
        batch_size=1, 
        padded_shapes=([padded_length, 6], [])  # Pads the time steps dimension to `padded_length`
    )

    return padded_data

def lstm_builder(num_classes):
    """
    Build an LSTM model for activity recognition.
        
    Parameters:
    - num_classes: The number of unique activity classes.
    """
    model = LSTMModel(num_classes=num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_dataset, number_of_training, val_dataset, number_of_validation, epochs=10):
    # Train the model on the training dataset and validate on the validation dataset
    VALIDATION_STEPS = number_of_validation // epochs
    STEPS_PER_EPOCH = number_of_training // epochs

    print('#######################')
    print('Training the model.....')
    print('#######################')
    history = model.fit(train_dataset, validation_data=val_dataset, 
              validation_steps=VALIDATION_STEPS, 
              epochs=epochs, 
              steps_per_epoch=STEPS_PER_EPOCH)  # Adjust epochs as needed
    print('#######################')
    print('Training completed.....')
    print('#######################')
    return history

def evaluate_model(model, test_dataset, number_of_eval_sample):
    """
    Evaluate the model on the test dataset.
    
    Parameters:
    - model: The trained model to evaluate.
    - test_dataset: The test dataset to evaluate the model on.
    """
    # Evaluate the model on the test dataset
    print('#######################')
    print('Evaluating the model...')
    print('#######################')
    eval_metrics = model.evaluate(test_dataset, steps=number_of_eval_sample, return_dict=True)
    print('#######################')
    print('Evaluation completed...')
    print('#######################')
    return eval_metrics

def get_kfold_splits(dataset_size, k=5):
    """
    Generates training and validation indices using KFold.
    
    Parameters:
    - dataset_size: The total number of samples in the dataset.
    - k: Number of folds.
    
    Returns:
    - A KFold object to split data for cross-validation.
    """
    kf = KFold(n_splits=k, shuffle=True)
    return kf.split(range(dataset_size))

def kfold_cross_validation(train_data_catalog, dataset_size, k=5, padding_length=0, num_classes=0, hyperparameters={}):
    """
    Perform K-Fold cross-validation with lazy loading using a generator and tf.data.Dataset.
    
    Parameters:
    - train_data_catalog: DataFrame with metadata to load the data.
    - dataset_size: The total number of samples in the dataset.
    - k: Number of folds.
    - padding_length: The length to pad the time steps to.
    - hyperparameters: Dictionary of hyperparameters to use for training.
    """
    if padding_length == 0:
        raise ValueError('Padding length must be greater than 0.')
    if num_classes == 0:
        raise ValueError('Number of classes must be greater than 0.')
    
    epochs = hyperparameters.get('epochs', 2)
    kfold = get_kfold_splits(dataset_size, k)
    fold_no = 1
    results = []

    for train_idx, val_idx in kfold:
        print(f'Fold {fold_no}...')
        number_of_training = len(train_idx)
        number_of_validation = len(val_idx)

        # Create training and validation datasets
        train_dataset = create_tf_dataset(train_data_catalog, RAW_DATA_PATH, train_idx, padding_length)
        val_dataset = create_tf_dataset(train_data_catalog, RAW_DATA_PATH, val_idx, padding_length)

        model = lstm_builder(num_classes)
        history = train_model(model, 
                              train_dataset, number_of_training, 
                              val_dataset, number_of_validation, 
                              epochs=epochs)

        # Evaluate the model
        eval_metrics = evaluate_model(model, val_dataset, number_of_validation)

        results.append({
            'fold': fold_no,
            'history': history.history,  # Store training history (accuracy, loss per epoch)
            'validation_loss': eval_metrics['loss'],
            'validation_accuracy': eval_metrics['accuracy']
        })
        fold_no += 1

    return results

def hyperparameter_tuning():
    # TODO: Implement hyperparameter tuning
    best_hyperparameters = None
    return best_hyperparameters

def main() -> None:
    # Initialize the DataCatalogProcessor
    data_catalog_processor = DataCatalogProcessor(DATA_PATH, RAW_DATA_PATH)
    data_catalog = prepare_data_catalog(data_catalog_processor)
    max_padding = data_catalog['window_size'].max()
    num_classes = data_catalog['activity_id'].nunique()  # Number of unique activity classes
    train_data_catalog, test_data_catalog = train_test_split(data_catalog, test_size=0.2, random_state=42)
    

    hyperparameters = {'epochs': 10}
    res = kfold_cross_validation(train_data_catalog, 
                                 dataset_size=100, k=5, 
                                 padding_length=max_padding,
                                 num_classes=num_classes,
                                 hyperparameters=hyperparameters)
    print(res)

    #  = hyperparameter_tuning()
    # print(f'Best hyperparameters: {best_hyperparameters}')
    # final_model = lstm_builder()
    # history = train_model(final_model, train_dataset, val_dataset, hyperparameters=best_hyperparameters)
    # eval_metrics = evaluate_model(final_model, test_dataset)

if __name__ == '__main__':
    main()
