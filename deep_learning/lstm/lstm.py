import tensorflow as tf

class LSTMModel(tf.keras.Model):
    """
    A custom LSTM-based model for sequence classification tasks.

    This model includes:
    - Masking to ignore padded values in variable-length sequences.
    - Layer normalization for stabilizing the inputs to the LSTM layer.
    - An LSTM layer for processing sequential data.
    - A Dense layer with ReLU activation.
    - Dropout for regularization.
    - An output Dense layer with softmax activation for classification.
    """
    def __init__(self, num_classes, lstm_units, dropout_rate):
        """
        Initialize the LSTMModel.

        Parameters:
        - num_classes (int): The number of output classes.
        - lstm_units (int): The number of units in the LSTM layer.
        - dropout_rate (float): The dropout rate for regularization.
        """
        super().__init__()
        self.masking = tf.keras.layers.Masking(mask_value=0.0)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.lstm = tf.keras.layers.LSTM(lstm_units)
        self.dense = tf.keras.layers.Dense(lstm_units // 2, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
            
    def call(self, inputs):
        """
        Forward pass of the model.

        Parameters:
        - inputs (Tensor): Input tensor of shape (batch_size, time_steps, features).

        Returns:
        - Tensor: Output predictions with shape (batch_size, num_classes).
        """
        x = self.masking(inputs)
        x = self.layer_norm(x)
        x = self.lstm(x)
        x = self.dense(x)
        x = self.dropout(x)
        output = self.fc(x)
        return output