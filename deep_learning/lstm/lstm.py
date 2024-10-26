import tensorflow as tf

class LSTMModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.masking = tf.keras.layers.Masking(mask_value=0.0)  # Mask padded values
        self.lstm = tf.keras.layers.LSTM(32)
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        x = self.masking(x)
        x = self.lstm(x)
        return self.fc(x)

    def summary(self):
        super().summary()
