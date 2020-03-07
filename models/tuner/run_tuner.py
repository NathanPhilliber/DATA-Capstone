from utils import *
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, LSTM, TimeDistributed, MaxPooling2D, BatchNormalization, Dropout, \
    Conv1D, MaxPooling1D, Bidirectional, CuDNNGRU, Reshape, Concatenate, concatenate, Input, CuDNNLSTM
from keras.optimizers import SGD, Adam
from models.networks.abstract_models.attention import Attention
from models.networks.abstract_models.base_model import BaseModel
from hyperas.distributions import uniform, normal, choice
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from models.spectra_preprocessor import SpectraPreprocessor
from datagen.spectra_loader import SpectraLoader


def model(X_train, y_train, X_test, y_test):
    num_channels = 10
    num_timesteps = 1001
    output_shape = 5
    model = Sequential()
    model.add(BatchNormalization(momentum={{uniform(0, 1)}}, input_shape=(num_timesteps, num_channels)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Attention(num_timesteps))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(500, activation='elu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, y_train, batch_size={{choice([16, 32, 128])}}, epochs=1, validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, verbose=1)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def data():
    spectra_pp = SpectraPreprocessor(dataset_name='set_01', use_generator=False)
    X_train, y_train, X_test, y_test = spectra_pp.transform(encoded=True)
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

