# %%
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import AUC, F1Score
import ray
from ray import tune
from ray.tune import Tuner
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import datetime

# Import your custom modules
from raw_data_preprocess import DataCatalogProcessor, DataLoader
from lstm import LSTMModel
from sklearn.model_selection import KFold

# %%
# Define paths to data
# DATA_PATH = '../../data'
# RAW_DATA_PATH = os.path.join(DATA_PATH, 'RawData')
# LOG_DIR = './logs/'
# os.makedirs(LOG_DIR, exist_ok=True)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data'))
RAW_DATA_PATH = os.path.join(DATA_PATH, 'RawData')
# LOG_DIR = os.path.join(SCRIPT_DIR, './logs/')
# os.makedirs(LOG_DIR, exist_ok=True)

# %%
def prepare_data_catalog(catalog: DataCatalogProcessor) -> pd.DataFrame:
    """
    Load the labels and activity data, merge them, and filter unwanted activities.

    Parameters:
    - catalog: DataCatalogProcessor object specifying the data to be loaded.
    """
    try:
        labels = catalog.load_labels()
        activity_labels = catalog.load_activity_labels()
        # print(labels)
        # print(activity_labels)
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

def create_tf_dataset(data_catalog, path, indices, padded_length, num_classes, batch_size) -> tf.data.Dataset:
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
    data = data.shuffle(buffer_size=300)
    # One-hot encode the labels
    data = data.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))
    # Pad and batch the dataset
    padded_data = data.padded_batch(
        batch_size=batch_size, 
        padded_shapes=([padded_length, 6], [num_classes])  # Pads the time steps dimension to `padded_length`
    )

    return padded_data.repeat()

def lstm_builder(num_classes, lstm_units, dropout_rate):
    """
    Build an LSTM model for activity recognition.
        
    Parameters:
    - num_classes: The number of unique activity classes.
    """
    model = LSTMModel(num_classes, lstm_units, dropout_rate)
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc', curve="ROC", multi_label=False),  # AUC for ROC
            F1Score(average='micro', name='f1_micro'),  # F1 Score micro
        ]
    )
    return model

class TensorBoardLogger(tf.keras.callbacks.Callback):
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

    # Re-define the paths inside the function
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data'))
    RAW_DATA_PATH = os.path.join(DATA_PATH, 'RawData')
    # LOG_DIR = os.path.join(SCRIPT_DIR, './logs/')
    # os.makedirs(LOG_DIR, exist_ok=True)

    # Prepare data
    data_catalog_processor = DataCatalogProcessor(DATA_PATH, RAW_DATA_PATH)
    data_catalog = prepare_data_catalog(data_catalog_processor)
    max_padding = data_catalog['window_size'].max()
    num_classes = data_catalog['activity_id'].nunique()
    label_encoder = LabelEncoder()
    data_catalog['encoded_labels'] = label_encoder.fit_transform(data_catalog['activity_id'])
    train_data_catalog, _ = train_test_split(data_catalog, test_size=0.2, random_state=31, stratify=data_catalog['encoded_labels'])

    epochs = config['epochs']
    lstm_units = config['lstm_units']
    dropout_rate = config['dropout_rate']
    batch_size = config['batch_size']
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)
    results = []

    # Get the trial directory from Ray Tune
    trial_dir = session.get_trial_dir()
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data_catalog)):
        print(f'Fold {fold + 1}...')
        train_dataset = create_tf_dataset(train_data_catalog, RAW_DATA_PATH, train_idx, max_padding, num_classes, batch_size)
        val_dataset = create_tf_dataset(train_data_catalog, RAW_DATA_PATH, val_idx, max_padding, num_classes, batch_size)
        model = lstm_builder(num_classes, lstm_units, dropout_rate)
        
        # Create a TensorBoard writer for this fold
        log_dir = os.path.join(trial_dir, f"fold_{fold+1}")
        writer = tf.summary.create_file_writer(log_dir)
        # Initialize metrics
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # Train for one epoch
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                steps_per_epoch=len(train_idx) // batch_size,
                validation_steps=len(val_idx) // batch_size,
                epochs=1,
                verbose=0,
                callbacks=[TensorBoardLogger(writer, fold=fold + 1)]
            )

            # Get metrics
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

# %%
def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    search_space = {
    'lstm_units': tune.randint(32, 129),  # Upper bound is exclusive
    'dropout_rate': tune.uniform(0.2, 0.7),
    'epochs': tune.choice([75]),
    'batch_size': tune.randint(8, 65)
    }

    algo = OptunaSearch(
        metric="val_loss",
        mode="min",
        seed=31
    )

    max_epochs = max(search_space['epochs'].categories)

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
            storage_path=os.path.abspath("./results/exp"),
            name=f"tune_{timestamp}"
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result("val_loss", mode="min")
    print("Best hyperparameters found were: ", best_result.config)

# %%
if __name__ == '__main__':
    main()
# def train_model(model, train_dataset, number_of_training, val_dataset, number_of_validation, batch_size, epochs=10, fold=1):
#     VALIDATION_STEPS = number_of_validation // batch_size
#     STEPS_PER_EPOCH = number_of_training // batch_size
    
#     # Create a TensorBoard writer for this fold
#     writer = tf.summary.create_file_writer(f"{LOG_DIR}/fold_{fold}")
#     history = model.fit(
#         train_dataset, validation_data=val_dataset,
#         epochs=epochs, steps_per_epoch=STEPS_PER_EPOCH,
#         validation_steps=VALIDATION_STEPS,
#         callbacks=[TensorBoardLogger(writer, fold)]
#     )
#     writer.close()
#     return history

# class TensorBoardLogger(tf.keras.callbacks.Callback):
#     def __init__(self, writer, fold):
#         super(TensorBoardLogger, self).__init__()
#         self.writer = writer
#         self.fold = fold

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         with self.writer.as_default():
#             for metric_name, metric_value in logs.items():
#                 tf.summary.scalar(f"fold_{self.fold}/{metric_name}", metric_value, step=epoch)

# def kfold_cross_validation(train_data_catalog, dataset_size, k=5, padding_length=0, num_classes=0, config={}):
#     epochs = config['epochs']
#     lstm_units = config['lstm_units']
#     dropout_rate = config['dropout_rate']
#     batch_size = config['batch_size']
#     kfold = KFold(n_splits=k, shuffle=True, random_state=123)
#     results = []

#     for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data_catalog)):
#         print(f'Fold {fold + 1}...')
#         train_dataset = create_tf_dataset(train_data_catalog, RAW_DATA_PATH, train_idx, padding_length, num_classes, batch_size)
#         val_dataset = create_tf_dataset(train_data_catalog, RAW_DATA_PATH, val_idx, padding_length, num_classes, batch_size)
#         model = lstm_builder(num_classes, lstm_units, dropout_rate)
        
#         history = train_model(model, train_dataset, len(train_idx), val_dataset, len(val_idx), batch_size, epochs=epochs, fold=fold + 1)
        
#         results.append({
#             'fold': fold + 1,
#             'train_loss': history.history['loss'][-1],
#             'train_accuracy': history.history['accuracy'][-1],
#             'validation_loss': history.history['val_loss'][-1],
#             'validation_accuracy': history.history['val_accuracy'][-1],
#             'train_auc': history.history['auc'][-1],
#             'validation_auc': history.history['val_auc'][-1],
#             'train_f1': history.history['f1_micro'][-1],
#             'validation_f1': history.history['val_f1_micro'][-1]
#         })

#     return results

# def hyperparameter_tuning(train_data_catalog, num_classes, max_padding):
#     search_space = [
#         {'lstm_units': 32, 'dropout_rate': 0.2, 'epochs': 10, 'batch_size': 32},
#         {'lstm_units': 64, 'dropout_rate': 0.3, 'epochs': 10, 'batch_size': 16},
#         {'lstm_units': 128, 'dropout_rate': 0.4, 'epochs': 10, 'batch_size': 8}
#     ]
#     best_config = None
#     best_val_loss = float('inf')

#     for config in search_space:
#         print(f"Testing configuration: {config}")
#         results = kfold_cross_validation(
#             train_data_catalog, =len(train_data_catalog),
#             k=5, padding_length=max_padding, num_classes=num_classes,
#             config=config
#         )

#         avg_val_loss = np.mean([result['validation_loss'] for result in results])
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_config = config

#         # Log each configuration's results for tensorboard logging later
#         for result in results:
#             print(f"Fold {result['fold']} - "
#                   f"Train Loss: {result['train_loss']:.4f}, Train Acc: {result['train_accuracy']:.4f}, "
#                   f"Val Loss: {result['validation_loss']:.4f}, Val Acc: {result['validation_accuracy']:.4f}, "
#                   f"Train AUC: {result['train_auc']:.4f}, Val AUC: {result['validation_auc']:.4f}, "
#                   f"Train F1: {result['train_f1']:.4f}, Val F1: {result['validation_f1']:.4f}")

#     print(f"Best configuration found: {best_config} with validation loss: {best_val_loss}")
#     return best_config

# def main():
#     data_catalog_processor = DataCatalogProcessor(DATA_PATH, RAW_DATA_PATH)
#     data_catalog = prepare_data_catalog(data_catalog_processor)
#     max_padding = data_catalog['window_size'].max()
#     num_classes = data_catalog['activity_id'].nunique()
#     label_encoder = LabelEncoder()
#     data_catalog['encoded_labels'] = label_encoder.fit_transform(data_catalog['activity_id'])
#     train_data_catalog, _ = train_test_split(data_catalog, test_size=0.2, random_state=31, stratify=data_catalog['encoded_labels'])
    
#     best_hyperparameters = hyperparameter_tuning(train_data_catalog, num_classes, max_padding)
#     print("Best Hyperparameters:", best_hyperparameters)
# # %%
# if __name__ == '__main__':
#     main()

# %%
# def evaluate_model(model, test_dataset, number_of_eval_sample):
#     """
#     Evaluate the model on the test dataset.
    
#     Parameters:
#     - model: The trained model to evaluate.
#     - test_dataset: The test dataset to evaluate the model on.
#     """
#     # Evaluate the model on the test dataset
#     print('#######################')
#     print('Evaluating the model...')
#     print('#######################')
#     eval_metrics = model.evaluate(test_dataset, return_dict=True)
#     print('#######################')
#     print('Evaluation completed...')
#     print('#######################')
#     return eval_metrics
# def kfold_cross_validation(train_data_catalog, dataset_size, k=5, padding_length=0, num_classes=0, config={}):
#     """
#     Perform K-Fold cross-validation with lazy loading using a generator and tf.data.Dataset.
    
#     Parameters:
#     - train_data_catalog: DataFrame with metadata to load the data.
#     - dataset_size: The total number of samples in the dataset.
#     - k: Number of folds.
#     - padding_length: The length to pad the time steps to.
#     - hyperparameters: Dictionary of hyperparameters to use for training.
#     """
#     if padding_length == 0:
#         raise ValueError('Padding length must be greater than 0.')
#     if num_classes == 0:
#         raise ValueError('Number of classes must be greater than 0.')
    
#     epochs = config['epochs']
#     lstm_units = config['lstm_units']
#     dropout_rate = config['dropout_rate']
#     kfold = get_kfold_splits(dataset_size, k)
#     fold_no = 1
#     results = []

#     for train_idx, val_idx in kfold:
#         print(f'Fold {fold_no}...')
#         number_of_training = len(train_idx)
#         number_of_validation = len(val_idx)
#         print(f'Number of training samples: {number_of_training}')
#         # Create training and validation datasets
#         train_dataset = create_tf_dataset(train_data_catalog, RAW_DATA_PATH, train_idx, padding_length, num_classes)
#         val_dataset = create_tf_dataset(train_data_catalog, RAW_DATA_PATH, val_idx, padding_length, num_classes)

#         model = lstm_builder(num_classes, lstm_units=64, dropout_rate=0.5)
#         history = train_model(model, 
#                               train_dataset, number_of_training, 
#                               val_dataset, number_of_validation, 
#                               epochs=epochs)

#         # Extract training and validation loss and accuracy from the training history
#         train_loss = history.history['loss'][-1]  
#         train_accuracy = history.history['accuracy'][-1]  
#         val_loss = history.history['val_loss'][-1]  
#         val_accuracy = history.history['val_accuracy'][-1]  
#         train_auc = history.history['auc'][-1]  
#         val_auc = history.history['val_auc'][-1]
#         train_f1 = history.history['f1_micro'][-1]
#         val_f1 = history.history['val_f1_micro'][-1]
#         results.append({
#             'fold': fold_no,
#             'history': history.history,  # Store training history (accuracy, loss per epoch)
#             'train_loss': train_loss,
#             'train_accuracy': train_accuracy,
#             'validation_loss': val_loss,
#             'validation_accuracy': val_accuracy,
#             'train_auc': train_auc,
#             'validation_auc': val_auc,
#             'train_f1': train_f1,
#             'validation_f1': val_f1
#         })
#         fold_no += 1

#     return results


# def main() -> None:
#     # Initialize the DataCatalogProcessor
#     data_catalog_processor = DataCatalogProcessor(DATA_PATH, RAW_DATA_PATH)
#     data_catalog = prepare_data_catalog(data_catalog_processor)
#     max_padding = data_catalog['window_size'].max()
#     num_classes = data_catalog['activity_id'].nunique()  # Number of unique activity classes
#     label_encoder = LabelEncoder()
#     data_catalog['encoded_labels'] = label_encoder.fit_transform(data_catalog['activity_id'])
#     train_data_catalog, test_data_catalog = train_test_split(data_catalog, test_size=0.2, random_state=31, stratify=data_catalog['encoded_labels'])
#     print(f"Shape of Train data catalog: {train_data_catalog.shape}")

#     config = {'epochs': 2,
#                        'lstm_units': 64,
#                        'dropout_rate': 0.5} 
#     res = kfold_cross_validation(train_data_catalog, 
#                                  dataset_size=100, k=3, 
#                                  padding_length=max_padding,
#                                  num_classes=num_classes,
#                                  config=config)
#     print(res)

# def get_kfold_splits(dataset_size, k=5):
#     """
#     Generates training and validation indices using KFold.
    
#     Parameters:
#     - dataset_size: The total number of samples in the dataset.
#     - k: Number of folds.
    
#     Returns:
#     - A KFold object to split data for cross-validation.
#     """
#     kf = KFold(n_splits=k, shuffle=True)
#     return kf.split(range(dataset_size))