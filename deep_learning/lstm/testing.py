# %%
from test import *
from tensorflow.keras.models import load_model
from lstm import LSTMModel
from main import *
from tensorflow.keras.layers import Layer
from test import *

# %%
# Load the model

def testing():
    # Load configuration parameters
    config = load_config(CONFIG_PATH, "config.yaml")
    epochs = config['epochs']
    lstm_units = config['lstm_units']
    dropout_rate = config['dropout_rate']
    batch_size = config['batch_size']
    print(batch_size)
    learning_rate = config['learning_rate']
    # Prepare data catalog
    data_catalog_processor = DataCatalogProcessor(DATA_PATH, RAW_DATA_PATH)
    data_catalog = prepare_data_catalog(data_catalog_processor, filter_unwanted=False)
    data_catalog = data_catalog.reset_index(drop=True)

    # Determine maximum padding length and number of classes
    max_padding = data_catalog['window_size'].max()
    num_classes = data_catalog['activity_id'].nunique()
    print(max_padding)
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

    model_path = './model/lstm_model_12.keras'
    model = load_model(model_path, custom_objects={'LSTMModel': LSTMModel})
    eval_metrics = eval_lstm(model, test_dataset, len(test_idx), batch_size, num_classes)
if __name__ == '__main__':
    testing()
# Now the model is loaded and ready to be used for predictions or further training
# %%
