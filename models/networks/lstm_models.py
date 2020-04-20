from utils import *
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, LSTM, TimeDistributed, MaxPooling2D, BatchNormalization, Dropout, \
    Conv1D, Bidirectional
from keras.layers import CuDNNGRU, CuDNNLSTM
from models.networks.abstract_models.attention import Attention
from models.networks.abstract_models.base_model import BaseModel


class GRUModel1(BaseModel):

    def set_params_range(self):
        return  {'momentum': {'type': 'float', 'min':0, 'max': 1, 'default': 0.98},
                  'gru_size_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                  'gru_size_2': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                  'dropout_1': {'type': 'float', 'min':0, 'max': 1, 'default': 0.5},
                  'dense_size': {'type': 'integer', 'min': 10, 'max': 800, 'default': 500},
                  'dropout_2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5}
                }

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        model = Sequential()
        model.add(BatchNormalization(params['momentum'], input_shape=(num_timesteps, num_channels)))
        model.add(Bidirectional(CuDNNGRU(params['gru_size_1'], return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(params['gru_size_2'], return_sequences=True)))
        model.add(Attention(num_timesteps))
        model.add(Dropout(params['dropout_1']))
        model.add(Dense(params['dense_size'], activation='elu'))
        model.add(Dropout(params['dropout_2']))
        model.add(Dense(output_shape, activation='softmax'))

        return model


class LSTMModel1(BaseModel):

    def set_params_range(self):
        return  {'momentum': {'type': 'float', 'min':0, 'max': 1, 'default': 0.98},
                  'lstm_size_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                  'lstm_size_2': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                  'dropout_1': {'type': 'float', 'min':0, 'max': 1, 'default': 0.5},
                  'dense_size': {'type': 'integer', 'min': 10, 'max': 800, 'default': 500},
                  'dropout_2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5}
                }

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        model = Sequential()
        model.add(BatchNormalization(momentum=params['momentum'], input_shape=(num_timesteps, num_channels)))
        model.add(Bidirectional(CuDNNLSTM(params['lstm_size_1'], return_sequences=True)))
        model.add(Bidirectional(CuDNNLSTM(params['lstm_size_2'], return_sequences=True)))
        model.add(Attention(num_timesteps))
        model.add(Dropout(params['dropout_1']))
        model.add(Dense(params['dense_size'], activation='elu'))
        model.add(Dropout(params['dropout_2']))
        model.add(Dense(output_shape, activation='softmax'))

        return model


class LSTMModelCPU(BaseModel):
    def set_params_range(self):
        return  {'momentum': {'type': 'float', 'min':0, 'max': 1, 'default': 0.98},
                  'lstm_size_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                  'lstm_size_2': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                  'dropout_1': {'type': 'float', 'min':0, 'max': 1, 'default': 0.5},
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
        return {'momentum': {'type': 'float', 'min':0, 'max': 1, 'default': 0.9},
                  'lstm_size_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                  'lstm_size_2': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                  'dropout_1': {'type': 'float', 'min':0, 'max': 1, 'default': 0.5},
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


class GoogleModel(BaseModel):

    def set_params_range(self):
        return {'conv_1': {'type': 'integer', 'min': 8, 'max': 64, 'default': 16},
                'conv_2': {'type': 'integer', 'min': 8, 'max': 64, 'default': 32},
                'bi_1': {'type': 'float', 'min': 8, 'max': 128, 'default': 128},
                'bi_2': {'type': 'integer', 'min': 8, 'max': 128, 'default': 128},
                'drop_1': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.01},
                'dense_1': {'type': 'float', 'min': 8, 'max': 128, 'default': 64},
                'drop_2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.05}
                }

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        """
        From: https://github.com/douglas125/SpeechCmdRecognition/blob/master/SpeechModels.py

        """
        num_attention = num_timesteps - 8
        model = Sequential()
        model.add(Conv1D(params['conv_1'], 5, input_shape=(num_timesteps, num_channels)))
        model.add(BatchNormalization())
        model.add(Conv1D(params['conv_2'], 5))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(params['bi_1'], return_sequences=True)))
        model.add(Bidirectional(LSTM(params['bi_2'], return_sequences=True)))
        model.add(Attention(num_attention))
        model.add(Dropout(params['drop_1']))
        model.add(Dense(params['dense_1'], activation='elu'))
        model.add(Dropout(params['drop_2']))
        model.add(Dense(output_shape, activation='softmax'))
        return model

