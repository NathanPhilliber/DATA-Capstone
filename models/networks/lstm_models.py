from utils import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, TimeDistributed, MaxPooling2D, BatchNormalization, \
    Dropout, Conv1D, MaxPooling1D, Bidirectional, Reshape, Concatenate, concatenate, Input, GRU
from tensorflow.keras.optimizers import SGD, Adam
from models.networks.abstract_models.attention import Attention
from models.networks.abstract_models.base_model import BaseModel


class GRUModel1(BaseModel):

    def set_params_range(self):
        return {'momentum': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.98},
                'gru_size_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'gru_size_2': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'dropout_1': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5},
                'dense_size': {'type': 'integer', 'min': 10, 'max': 800, 'default': 500},
                'dropout_2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5}
                }

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        model = Sequential()
        model.add(BatchNormalization(params['momentum'], input_shape=(num_timesteps, num_channels)))
        model.add(Bidirectional(GRU(params['gru_size_1'], return_sequences=True)))
        model.add(Bidirectional(GRU(params['gru_size_2'], return_sequences=True)))
        model.add(Attention(num_timesteps))
        model.add(Dropout(params['dropout_1']))
        model.add(Dense(params['dense_size'], activation='elu'))
        model.add(Dropout(params['dropout_2']))
        model.add(Dense(output_shape, activation='softmax'))

        return model


class LSTMModel1(BaseModel):

    def set_params_range(self):
        return {'momentum': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.98},
                'lstm_size_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'lstm_size_2': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'dropout_1': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5},
                'dense_size': {'type': 'integer', 'min': 10, 'max': 800, 'default': 500},
                'dropout_2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5}
                }

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        model = Sequential()
        model.add(BatchNormalization(momentum=params['momentum'], input_shape=(num_timesteps, num_channels)))
        model.add(Bidirectional(LSTM(params['lstm_size_1'], return_sequences=True)))
        model.add(Bidirectional(LSTM(params['lstm_size_2'], return_sequences=True)))
        model.add(Attention(num_timesteps))
        model.add(Dropout(params['dropout_1']))
        model.add(Dense(params['dense_size'], activation='elu'))
        model.add(Dropout(params['dropout_2']))
        model.add(Dense(output_shape, activation='softmax'))

        return model


class LSTMModelCPU(BaseModel):
    def set_params_range(self):
        return {'momentum': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.98},
                'lstm_size_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'lstm_size_2': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'dropout_1': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5},
                'dense_size': {'type': 'integer', 'min': 10, 'max': 800, 'default': 500},
                'dropout_2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5}
                }

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        model = Sequential()
        model.add(BatchNormalization(params['momentum'], input_shape=(num_timesteps, num_channels)))
        model.add(Bidirectional(LSTM(params['lstm_size_1'], return_sequences=True)))
        model.add(Bidirectional(LSTM(params['lstm_size_2'], return_sequences=True)))
        model.add(Attention(num_timesteps))
        model.add(Dropout(params['dropout_1']))
        model.add(Dense(params['dense_size'], activation='elu'))
        model.add(Dropout(params['dropout_2']))
        model.add(Dense(output_shape, activation='softmax'))

        return model


class LSTMModelCPU2(BaseModel):

    def set_params_range(self):
        return {'momentum': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.9},
                'lstm_size_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'lstm_size_2': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'dropout_1': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5},
                'dense_size': {'type': 'integer', 'min': 10, 'max': 800, 'default': 500},
                'dropout_2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5}
                }

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        model = Sequential()
        model.add(BatchNormalization(momentum=params['momentum'], input_shape=(num_timesteps, num_channels)))
        model.add(Bidirectional(LSTM(params['lstm_size_1'], return_sequences=True)))
        model.add(Bidirectional(LSTM(params['lstm_size_2'], return_sequences=True)))
        model.add(Attention(num_timesteps))
        model.add(Dropout(params['dropout_1']))
        model.add(Dense(params['dense_size'], activation='elu'))
        model.add(Dropout(params['dropout_2']))
        model.add(Dense(output_shape, activation='softmax'))

        return model
