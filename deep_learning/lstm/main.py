import os
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split


sys.path.append(os.path.abspath('../../scripts'))

from raw_data_preprocess import DataCatalog
from lstm import LSTMModel

# Define paths to data
DATA_PATH = '../../data'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'RawData')

def load_and_prepare_data(catalog: DataCatalog) -> pd.DataFrame:
    """
    Load the labels and activity data, merge them, and filter unwanted activities.
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

    # Shuffle data at the experiment level
    filtered_data = filtered_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return filtered_data

def normalize(data):
    """
    Normalize each feature in the dataset.
    """
    return (data - data.mean()) / data.std()

def create_padded_dataset(catalog, data, batch_size=5, padded_length=233) -> tf.data.Dataset:
    """
    Create a padded TensorFlow dataset from the filtered data.
    """
    data = catalog.load_experiment_data(data)

    # Apply normalization
    data = normalize(data)

    # Pad and batch the dataset
    padded_data = data.padded_batch(
        batch_size=batch_size, 
        padded_shapes=([padded_length, 6], [])  # Pads the time steps dimension to `padded_length`
    )

    return padded_data

def main() -> None:
    # Initialize the DataCatalog
    catalog = DataCatalog(DATA_PATH, RAW_DATA_PATH)

    # Load, filter, and prepare the data
    filtered_data = load_and_prepare_data(catalog)
    max_padding = filtered_data['window_size'].max()
    num_classes = filtered_data['activity_id'].nunique()  # Number of unique activity classes

    # Split into train, validation, and test sets (60% train, 20% validation, 20% test)
    train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

    # Create TensorFlow datasets for each split
    train_dataset = create_padded_dataset(catalog, train_data, batch_size=1, padded_length=max_padding)
    val_dataset = create_padded_dataset(catalog, val_data, batch_size=1, padded_length=max_padding)
    test_dataset = create_padded_dataset(catalog, test_data, batch_size=1, padded_length=max_padding)

    # Initialize the LSTM model
    model = LSTMModel(num_classes=num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the training dataset and validate on the validation dataset
    model.fit(train_dataset, validation_data=val_dataset, epochs=1)  # Adjust epochs as needed

    # Evaluate the model on the test dataset
    model.evaluate(test_dataset)

if __name__ == '__main__':
    main()
