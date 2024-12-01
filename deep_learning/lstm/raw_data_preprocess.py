import pandas as pd
import os
import numpy as np
import tensorflow as tf

class DataCatalogProcessor:
    def __init__(self, data_path: str, raw_data_path: str):
        """
        Initializes the DataCatalog with paths to the data.
        
        Parameters:
        - data_path: Path to the main dataset (e.g., activity labels).
        - raw_data_path: Path to the raw accelerometer and gyroscope data.
        """
        self.data_path = data_path
        self.raw_data_path = raw_data_path

    def load_labels(self) -> pd.DataFrame:
        """
        Loads the labels dataset with experiment, user, activity, and start/end indices.
        
        Returns:
        - pd.DataFrame: The labels dataset.
        """
        labels_file = os.path.join(self.raw_data_path, 'labels.txt')
        names = ['exp_id', 'user_id', 'activity_id', 'start_index', 'end_index']
        return pd.read_csv(labels_file, sep='\s+', header=None, names=names)

    def load_activity_labels(self) -> pd.DataFrame:
        """
        Loads the activity labels dataset mapping activity IDs to activity names.
        
        Returns:
        - pd.DataFrame: The activity labels dataset.
        """
        activity_labels_file = os.path.join(self.data_path, 'activity_labels.txt')
        names = ['activity_id', 'activity_name']
        return pd.read_csv(activity_labels_file, sep='\s+', header=None, names=names)
    
    def merge_labels_with_activity(self, labels: pd.DataFrame, activity_labels: pd.DataFrame) -> pd.DataFrame:
        """Merge labels with activity names for better analysis."""
        return labels.merge(activity_labels, on='activity_id', how='left')

    def compute_window_size(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Compute window size by subtracting start_index from end_index."""
        labels['window_size'] = labels['end_index'] - labels['start_index']
        return labels

    def drop_activity(self, labels: pd.DataFrame, unwanted_activities: list) -> pd.DataFrame:
        """Drop rows from the labels DataFrame that have unwanted activities."""
        return labels[~labels['activity_name'].isin(unwanted_activities)]

    def display_activity_statistics(self, labels: pd.DataFrame) -> None:
        """Display descriptive statistics of activity durations."""
        print(labels.groupby('activity_name')['window_size'].describe().sort_values(by='mean', ascending=True))

class DataSet:
    def __init__(self, data_catalog: pd.DataFrame, raw_data_path: str):
        """
        Initializes the DataSet with the catalog of data and raw data path.
        
        Parameters:
        - data_catalog: DataFrame containing all experiment metadata.
        - raw_data_path: Path to the raw sensor data (accelerometer and gyroscope).
        """
        self.data_catalog = data_catalog
        self.raw_data_path = raw_data_path

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file into a Pandas DataFrame.
        
        Parameters:
        - file_path: Path to the file.
        
        Returns:
        - pd.DataFrame: Loaded data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path, sep='\s+', header=None)

    def _load_sensor_data(self, exp_id: int, user_id: int) -> (pd.DataFrame, pd.DataFrame):
        """
        Load accelerometer and gyroscope data for a given experiment and user.
        
        Parameters:
        - exp_id: The experiment ID.
        - user_id: The user ID.
        
        Returns:
        - (pd.DataFrame, pd.DataFrame): Tuple containing accelerometer and gyroscope data.
        """
        acc_file = os.path.join(self.raw_data_path, f"acc_exp{exp_id:02d}_user{user_id:02d}.txt")
        gyro_file = os.path.join(self.raw_data_path, f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt")
        
        acc_data = self._load_file(acc_file)
        gyro_data = self._load_file(gyro_file)
        return acc_data, gyro_data

    def _extract_window(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame, start_idx: int, end_idx: int) -> np.array:
        """
        Extract a time window from the accelerometer and gyroscope data.
        
        Parameters:
        - acc_data: Accelerometer data as a DataFrame.
        - gyro_data: Gyroscope data as a DataFrame.
        - start_idx: Start index of the window.
        - end_idx: End index of the window.
        
        Returns:
        - np.array: Concatenated accelerometer and gyroscope data for the window.
        """
        acc_window = acc_data.iloc[start_idx:end_idx].values
        gyro_window = gyro_data.iloc[start_idx:end_idx].values
        return np.concatenate((acc_window, gyro_window), axis=1)

    def __getitem__(self, index: int) -> (np.array, int):
        """
        Retrieve a sample (data window and label) for a given index.
        
        Parameters:
        - index: The index of the sample in the data catalog.
        
        Returns:
        - np.array: A data window.
        - int: The activity label corresponding to the window.
        """
        row = self.data_catalog.iloc[index]
        exp_id = row['exp_id']
        user_id = row['user_id']
        activity_id = row['activity_id'] - 1  # Zero-based label
        start_idx = row['start_index']
        end_idx = row['end_index']
        
        # Load sensor data (accelerometer and gyroscope)
        acc_data, gyro_data = self._load_sensor_data(exp_id, user_id)
        
        # Extract the relevant window
        window = self._extract_window(acc_data, gyro_data, start_idx, end_idx)
        
        return window.astype(np.float32), activity_id

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
        - int: Number of samples in the dataset.
        """
        return len(self.data_catalog)

class DataLoader:
    def __init__(self, data_catalog: pd.DataFrame, raw_data_path: str):
        """
        DataLoader to handle loading data in KerasTensor format without batching or shuffling.
        
        Parameters:
        - data_catalog: DataCatalog object specifying the data to be loaded.
        """
        self.data_catalog = data_catalog
        self.dataset = DataSet(data_catalog, raw_data_path)  # Load the dataset

    def _data_generator(self, indices):
        """
        Generator that yields individual samples of data based on the provided indices.
        
        Parameters:
        - indices: List or array of indices to load the data.
        """
        for idx in indices:
            yield self.dataset[idx]  # Yield individual samples based on the indices

    def create_keras_tensor(self, indices):
        """
        Create a TensorFlow Dataset from the data generator using specific indices.
        
        Parameters:
        - indices: List or array of indices to load the data.
        
        Returns:
        - tf.data.Dataset: TensorFlow Dataset ready for model training or evaluation.
        """
        output_signature = (
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32),  # Time-series data with 6 features
            tf.TensorSpec(shape=(), dtype=tf.int32)            # Activity labels
        )
        
        tf_dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(indices),
            output_signature=output_signature
        )
        
        return tf_dataset

    def load_experiment_data(self, indices):
        """
        Load the data using the DataLoader for specific indices.
        
        Parameters:
        - indices: List or array of indices to load the data.
        
        Returns:
        - tf.data.Dataset: TensorFlow Dataset ready for model training or evaluation.
        """
        return self.create_keras_tensor(indices)  # Use the generator to create the dataset