from utils import *
from datagen.SpectraGenerator import SpectraGenerator
from models.SpectraPreprocessor import SpectraPreprocessor
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, LSTM, TimeDistributed, MaxPooling2D, BatchNormalization, Dropout, \
    Conv1D, MaxPooling1D, Bidirectional, CuDNNGRU, Reshape, Concatenate, concatenate, Input, CuDNNLSTM
from keras.optimizers import SGD, Adam
from models.attention import Attention
from models.BaseModel import BaseModel


class LstmModel1(BaseModel):

    def build_model(self, num_channels, num_timesteps, output_shape):
        model = Sequential()
        model.add(BatchNormalization(momentum=0.98, input_shape=(num_timesteps, num_channels)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
        model.add(Attention(num_timesteps))
        model.add(Dropout(.5))
        model.add(Dense(500, activation='elu'))
        model.add(Dropout(.5))
        model.add(Dense(output_shape, activation='softmax'))

        return model
