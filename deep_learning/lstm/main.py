import os
import tensorflow as tf
import pandas as pd
import numpy as np
import sys

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

    return filtered_data

def create_padded_dataset(catalog, data, batch_size=5, padded_length=233) -> tf.data.Dataset:
    """
    Create a padded TensorFlow dataset from the filtered data.
    """
    data = catalog.load_experiment_data(data)

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
    filtered_data = load_and_prepare_data(catalog).head(10)
    max_padding = filtered_data['window_size'].max()

    # Number of unique activity classes
    num_classes = filtered_data['activity_id'].nunique()

    # Create the padded dataset
    padded_data = create_padded_dataset(catalog, filtered_data, batch_size=1, padded_length=max_padding)

    # # Initialize the LSTM model
    model = LSTMModel(num_classes=num_classes)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the padded dataset
    number_of_batches = len(filtered_data) # this depends on the number of training samples
    EPOCHS = 2 # user define epochs
    STEPS_PER_EPOCH = number_of_batches // EPOCHS # this is fixed, don't change
    model.fit(padded_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

    # Evaluate the model
    number_of_eval_sample = len(filtered_data) # this should be changed to the number of evaluation samples
    model.evaluate(padded_data, steps=number_of_eval_sample)

if __name__ == '__main__':
    main()
