import tensorflow as tf

class LSTMModel(tf.keras.Model):
    def __init__(self, num_classes, lstm_units, dropout_rate):
        super().__init__()
        self.masking = tf.keras.layers.Masking(mask_value=0.0)  # Mask padded values
        self.lstm = tf.keras.layers.LSTM(lstm_units)
        self.dense = tf.keras.layers.Dense(lstm_units// 2, activation='relu')  
        self.dropout = tf.keras.layers.Dropout(dropout_rate) 
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        x = self.masking(x)
        x = self.lstm(x)
        x = self.dense(x) 
        x = self.dropout(x)  
        return self.fc(x)

    def summary(self):
        super().summary()
