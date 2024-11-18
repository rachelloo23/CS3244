import os
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import AUC, F1Score

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

# Import custom modules (ensure these are available in your environment)
from raw_data_preprocess import DataCatalogProcessor, DataLoader
from lstm import LSTMModel

# Set up paths to data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data'))
RAW_DATA_PATH = os.path.join(DATA_PATH, 'RawData')


def prepare_data_catalog(catalog: DataCatalogProcessor) -> pd.DataFrame:
    """
    Load labels and activity data, compute window sizes, merge with activity names,
    and filter out unwanted activities.

    Parameters:
    - catalog: DataCatalogProcessor object specifying the data to be loaded.

    Returns:
    - filtered_data: A DataFrame containing the processed and filtered data catalog.
    """
    try:
        labels = catalog.load_labels()
        activity_labels = catalog.load_activity_labels()
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        exit(1)

    # Compute window sizes (window_size = end_time - start_time) and sort
    labels = catalog.compute_window_size(labels)
    labels.sort_values(by='window_size', ascending=True, inplace=True)

    # Merge labels with activity names
    labels_with_activity = catalog.merge_labels_with_activity(labels, activity_labels)

    # Filter out unwanted activities (e.g., those containing 'TO')
    unwanted_activities = activity_labels[
        activity_labels['activity_name'].str.contains('TO')
    ]['activity_name'].tolist()
    filtered_data = catalog.drop_activity(labels_with_activity, unwanted_activities)

    return filtered_data


def create_tf_dataset(
    data_catalog, path, indices, padded_length, num_classes, batch_size
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from the data catalog using specified indices.

    Parameters:
    - data_catalog: DataFrame with metadata to load the data.
    - path: Path to the raw data files.
    - indices: The indices of the data to load.
    - padded_length: The length to pad the time steps to.
    - num_classes: Number of classes for one-hot encoding.
    - batch_size: The batch size for the dataset.

    Returns:
    - padded_data: The prepared TensorFlow dataset.
    """
    data_loader = DataLoader(data_catalog, path)
    data = data_loader.load_experiment_data(indices)
    data = data.shuffle(buffer_size=len(indices))

    # One-hot encode the labels
    data = data.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))

    # Pad and batch the dataset
    padded_data = data.padded_batch(
        batch_size=batch_size,
        padded_shapes=([padded_length, 6], [num_classes])  # Pads time steps to `padded_length`
    )

    return padded_data.repeat()


def lstm_builder(num_classes, lstm_units, dropout_rate):
    """
    Build and compile an LSTM model for activity recognition.

    Parameters:
    - num_classes: The number of unique activity classes.
    - lstm_units: Number of units in the LSTM layer.
    - dropout_rate: Dropout rate for regularization.

    Returns:
    - model: A compiled TensorFlow Keras model.
    """
    model = LSTMModel(num_classes, lstm_units, dropout_rate)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc', curve="ROC", multi_label=False),
            F1Score(average='micro', name='f1_micro'),
            F1Score(average='macro', name='f1_macro'),
            F1Score(average='weighted', name='f1_weighted'),
        ]
    )
    return model


class TensorBoardLogger(tf.keras.callbacks.Callback):
    """
    Custom callback to log metrics to TensorBoard for each fold during cross-validation.
    """
    def __init__(self, writer, fold):
        super(TensorBoardLogger, self).__init__()
        self.writer = writer
        self.fold = fold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            for metric_name, metric_value in logs.items():
                tf.summary.scalar(f"fold_{self.fold}/{metric_name}", metric_value, step=epoch)


def lstm_tune(config):
    """
    Objective function for hyperparameter tuning using Ray Tune.

    Parameters:
    - config: Dictionary containing hyperparameters to tune.
    """
    # Re-define the paths inside the function to ensure availability in Ray Tune workers
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data'))
    RAW_DATA_PATH = os.path.join(DATA_PATH, 'RawData')

    # Prepare data
    data_catalog_processor = DataCatalogProcessor(DATA_PATH, RAW_DATA_PATH)
    data_catalog = prepare_data_catalog(data_catalog_processor)
    max_padding = data_catalog['window_size'].max()
    num_classes = data_catalog['activity_id'].nunique()

    # Encode labels
    label_encoder = LabelEncoder()
    data_catalog['encoded_labels'] = label_encoder.fit_transform(data_catalog['activity_id'])

    # Split data into training and test sets
    train_data_catalog, _ = train_test_split(
        data_catalog,
        test_size=0.2,
        random_state=31,
        stratify=data_catalog['encoded_labels']
    )

    # Extract hyperparameters from config
    epochs = config['epochs']
    lstm_units = config['lstm_units']
    dropout_rate = config['dropout_rate']
    batch_size = config['batch_size']

    # Set up K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)

    # Get the trial directory from Ray Tune for logging
    trial_dir = session.get_trial_dir()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data_catalog)):
        print(f'Fold {fold + 1}...')

        # Create TensorFlow datasets for training and validation
        train_dataset = create_tf_dataset(
            train_data_catalog,
            RAW_DATA_PATH,
            train_idx,
            max_padding,
            num_classes,
            batch_size
        )
        val_dataset = create_tf_dataset(
            train_data_catalog,
            RAW_DATA_PATH,
            val_idx,
            max_padding,
            num_classes,
            batch_size
        )

        # Build and compile the model
        model = lstm_builder(num_classes, lstm_units, dropout_rate)

        # Create a TensorBoard writer for this fold
        log_dir = os.path.join(trial_dir, f"fold_{fold+1}")
        writer = tf.summary.create_file_writer(log_dir)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Train the model for one epoch
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                steps_per_epoch=len(train_idx) // batch_size,
                validation_steps=len(val_idx) // batch_size,
                epochs=1,
                verbose=0,
                callbacks=[TensorBoardLogger(writer, fold=fold + 1)]
            )

            # Get training and validation metrics
            metrics = history.history

            # Report metrics to Ray Tune
            report = {
                'train_loss': metrics['loss'][0],
                'train_accuracy': metrics['accuracy'][0],
                'val_loss': metrics['val_loss'][0],
                'val_accuracy': metrics['val_accuracy'][0],
                'train_auc': metrics['auc'][0],
                'val_auc': metrics['val_auc'][0],
                'train_f1_micro': metrics['f1_micro'][0],
                'val_f1_micro': metrics['val_f1_micro'][0],
                'epoch': epoch + 1,
                'fold': fold + 1
            }

            # Report the validation loss for Ray Tune to minimize
            session.report(report)

        writer.close()


def main():
    """
    Main function to perform hyperparameter tuning using Ray Tune and Optuna.
    """
    # Generate a timestamp for naming the tuning run
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Define the search space for hyperparameters
    search_space = {
        'lstm_units': tune.randint(32, 129),  # Upper bound is exclusive
        'dropout_rate': tune.uniform(0.2, 0.7),
        'epochs': tune.choice([75]),  # Using a fixed number of epochs
        'batch_size': tune.randint(8, 65)
    }

    # Set up the hyperparameter optimization algorithm (Optuna)
    algo = OptunaSearch(
        metric="val_loss",
        mode="min",
        seed=31
    )

    # Get the maximum number of epochs from the search space
    max_epochs = max(search_space['epochs'].categories)*10

    # Set up the Ray Tune Tuner
    tuner = tune.Tuner(
        lstm_tune,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            search_alg=algo,
            num_samples=100,
            scheduler=ASHAScheduler(
                time_attr='training_iteration',
                max_t=max_epochs,
                grace_period=5,
                reduction_factor=2
            ),
            # Optional: Specify resources per trial
            # resources_per_trial={"cpu": 2, "gpu": 0}
        ),
        param_space=search_space,
        run_config=ray.air.RunConfig(
            storage_path=os.path.abspath("./log/exp"),
            name=f"tune_{timestamp}"
        )
    )

    # Run the hyperparameter tuning
    results = tuner.fit()

    # Get the best result based on validation loss
    best_result = results.get_best_result("val_loss", mode="min")
    print("Best hyperparameters found were:", best_result.config)


if __name__ == '__main__':
    main()