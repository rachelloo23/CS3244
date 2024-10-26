import pandas as pd
import os
import numpy as np
import tensorflow as tf

class DataCatalog:
    def __init__(self, data_path: str, raw_data_path: str):
        """Initialize the DataCatalog with paths to the data."""
        self.data_path = data_path
        self.raw_data_path = raw_data_path

    def _load_data(self, fname: str, names: list) -> pd.DataFrame:
        """Load a dataset from a file with specified column names."""
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File not found: {fname}")
        return pd.read_csv(fname, sep='\s+', header=None, names=names)

    def load_labels(self) -> pd.DataFrame:
        """Load the labels dataset with experiment, user, activity, and start/end indices."""
        names = ['exp_id', 'user_id', 'activity_id', 'start_index', 'end_index']
        path = os.path.join(self.raw_data_path, 'labels.txt')
        return self._load_data(path, names)

    def load_activity_labels(self) -> pd.DataFrame:
        """Load the activity labels dataset mapping activity IDs to activity names."""
        names = ['activity_id', 'activity_name']
        path = os.path.join(self.data_path, 'activity_labels.txt')
        return self._load_data(path, names)

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

    def _load_raw_data(self, exp_id: int, user_id: int) -> (pd.DataFrame, pd.DataFrame):
        """
        Load accelerometer and gyroscope data for a specific experiment.
        
        Parameters:
        - exp_id: The experiment ID.
        - user_id: The user ID.
        
        Returns:
        - acc_data: The accelerometer data as a DataFrame.
        - gyro_data: The gyroscope data as a DataFrame.
        """
        acc_file = os.path.join(self.raw_data_path, f"acc_exp{exp_id:02d}_user{user_id:02d}.txt")
        gyro_file = os.path.join(self.raw_data_path, f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt")
        
        acc_data = pd.read_csv(acc_file, sep='\s+', header=None)
        gyro_data = pd.read_csv(gyro_file, sep='\s+', header=None)
        
        return acc_data, gyro_data

    def _extract_window(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame, start_idx: int, end_idx: int) -> np.array:
        """
        Extract the window of accelerometer and gyroscope data for a given start and end index.
        
        Parameters:
        - acc_data: The accelerometer data.
        - gyro_data: The gyroscope data.
        - start_idx: The starting index of the window.
        - end_idx: The ending index of the window.
        
        Returns:
        - window: A concatenated array of accelerometer and gyroscope data for the window.
        """
        acc_window = acc_data.iloc[start_idx:end_idx].values
        gyro_window = gyro_data.iloc[start_idx:end_idx].values
        
        # Concatenate accelerometer and gyroscope data along the features axis
        window = np.concatenate((acc_window, gyro_window), axis=1)
        
        return window

    def load_experiment_data(self, data_catalog: pd.DataFrame): 
        """
        Create a TensorFlow dataset from the provided data catalog.
        
        Parameters:
        - data_catalog: DataFrame containing all experiments and their metadata.
        
        Returns:
        - tf.data.Dataset: A TensorFlow Dataset ready for training.
        """
        dataset = tf.data.Dataset.from_generator(
            lambda: self._generator(data_catalog), 
            output_signature=(
                tf.TensorSpec(shape=(None, 6), dtype=tf.float32),  # Time-series data (6 features)
                tf.TensorSpec(shape=(), dtype=tf.int32)            # Activity labels (single label per window)
            )
        )
        
        return dataset

    def _generator(self, data_catalog: pd.DataFrame):
        """
        Generator function to yield windows of data and corresponding labels from the provided data catalog.
        """
        for _, row in data_catalog.iterrows():
            exp_id = row['exp_id']
            user_id = row['user_id']
            activity_id = row['activity_id']  # This is the target label (what you are predicting)
            start_idx = row['start_index']
            end_idx = row['end_index']
            
            # Load the raw accelerometer and gyroscope data for this experiment
            acc_data, gyro_data = self._load_raw_data(exp_id, user_id)
            
            # Extract the relevant window of data
            window = self._extract_window(acc_data, gyro_data, start_idx, end_idx)
            
            # Yield the window (features) and its corresponding label (activity_id)
            yield window.astype(np.float32), np.array(activity_id-1, dtype=np.int32)  # Make sure activity_id is a scalar integer

    

import tensorflow as tf

class LSTMModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.masking = tf.keras.layers.Masking(mask_value=0.0)  # Mask padded values
        # self.reshape = tf.keras.layers.Reshape((-1, 6))
        self.lstm = tf.keras.layers.LSTM(32)
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        print(x.shape)
        x = self.masking(x)
        print(x.shape)
        x = self.lstm(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        return x

    def summary(self):
        super().summary()
    

from tensorflow.keras.preprocessing.sequence import pad_sequences




if __name__ == '__main__':
    # Initialize the DataCatalog with the paths
    catalog = DataCatalog(DATA_PATH, RAW_DATA_PATH)

    # Load the labels and activity labels with error handling
    try:
        labels = catalog.load_labels()
        activity_labels = catalog.load_activity_labels()
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        exit(1)

    # Compute window sizes and sort by window size
    labels = catalog.compute_window_size(labels)
    labels.sort_values(by='window_size', ascending=True, inplace=True)

    # Merge labels with activity names for better analysis
    labels_with_activity = catalog.merge_labels_with_activity(labels, activity_labels)

    # Display descriptive statistics of activity durations
    print('\n###########################################')
    print('Activities Duration Descriptive Statistics:')
    print('###########################################')
    catalog.display_activity_statistics(labels_with_activity)

    # Filter unwanted activities that contain 'TO' in their names
    unwanted_activities = activity_labels[activity_labels['activity_name'].str.contains('TO')]['activity_name'].tolist()
    print('\n###########################################')
    print("Unwanted Activities (Transition Activities)")
    print('###########################################')
    print(unwanted_activities)

    # Drop unwanted activities
    labels_filtered = catalog.drop_activity(labels_with_activity, unwanted_activities)

    # Display descriptive statistics of activity durations after filtering
    print('\n###########################################')
    print('Filtered Activities Duration Descriptive Statistics:')
    print('###########################################')
    catalog.display_activity_statistics(labels_filtered)

    small_labels = labels_filtered.head(5)
    print(small_labels)
    res = catalog.load_experiment_data(small_labels)
    padded_res = res.padded_batch(
        # batch_size=10,  # Batch size (number of sample)
        padded_shapes=([233, 6], [])  # Pads the first dimension (time steps), keeps features (6) and labels the same
    )

    # print(padded_res.element_spec)

    # Number of unique activity classes in the dataset (modify based on actual number of activities)
    num_classes = len(small_labels['activity_id'].unique())

    # Initialize the LSTM model with the number of classes
    model = LSTMModel(num_classes=num_classes)
    # inputs = tf.keras.layers.Input(shape=(None,6))
    # model(inputs)

    # Display the model summary
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.summary()


    # Fit the model on the dataset for 1 epoch (for quick testing, increase epochs for real training)
    model.fit(padded_res, epochs=1)

    # Evaluate the model on the same dataset (for testing, normally you'd use a separate test set)
    model.evaluate(padded_res)