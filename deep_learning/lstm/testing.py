# %%
from test import *
from tensorflow.keras.models import load_model
from lstm import LSTMModel
from main import *
from tensorflow.keras.layers import Layer

# %%
# Load the model
model_path = './model/lstm_model.h5'
model = load_model(model_path, custom_objects={'LSTMModel': LSTMModel})
if __name__ == '__main__':
    testing()
# Now the model is loaded and ready to be used for predictions or further training