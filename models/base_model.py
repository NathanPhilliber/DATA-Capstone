from utils import *
from abc import ABC
from abc import abstractmethod
from datagen import Spectrum, SpectraLoader, SpectraGenerator
from sklearn.preprocessing import OneHotEncoder
import numpy as np
#import matplotlib.pyplot as plt
import json
import random


def get_model_params(model_name, train_path, test_path, model):
    model_dict = dict()
    model_dict['name'] = model_name
    model_dict['train_path'] = train_path
    model_dict['test_path'] = test_path
    model_dict['model_params'] = model.to_dict()
    return model_dict


def save_model(model_name, train_path, test_path, model):
    model_configs = get_model_params(model_name, train_path, test_path, model)
    model_path = os.path.join(MODEL_RES_DIR, model_name)
    with open(model_path, 'w') as f:
        json.dump(str(model_configs), f)

    print(f'Saved model results as {model_name} in the file: {model_path}')


class SpectraPreprocessor:

    def __init__(self, dataset_name, use_generator=False):
        self.train_spectra_loader = SpectraLoader(dataset_name=dataset_name, subset_prefix=TRAIN_DATASET_PREFIX, eval_now=not use_generator)
        self.test_spectra_loader = SpectraLoader(dataset_name=dataset_name, subset_prefix=TEST_DATASET_PREFIX)
        self.one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')

        _, y_test = self.get_data(self.test_spectra_loader)
        self.one_hot_encoder.fit(y_test)

        self.datagen_config = json.load(open(os.path.join(DATA_DIR, dataset_name, DATAGEN_CONFIG), "r"))

    def get_data(self, loader):
        dm = np.array(loader.get_dm())
        X = dm.reshape(dm.shape[0], dm.shape[2], dm.shape[1], 1)
        y = np.array(loader.get_n())
        y = y.reshape(y.shape[0], 1)
        return X, y

    def transform(self, encoded=False):
        return (*self.transform_train(encoded=encoded), *self.transform_test(encoded=encoded))

    def transform_train(self, encoded=False):
        X_train, y_train = self.get_data(self.train_spectra_loader)
        if encoded:
            y_train = self.one_hot_encoder.transform(y_train)
        return X_train, y_train

    def transform_test(self, encoded=False):
        print("Transforming test")
        X_test, y_test = self.get_data(self.test_spectra_loader)
        if encoded:
            y_test = self.one_hot_encoder.transform(y_test)
        return X_test, y_test

    def train_generator(self, batch_size, encoded=False):
        cur_set_i = 0
        files = self.train_spectra_loader.get_data_files()

        num_files = len(files)
        spectra_x = None
        spectra_y = None

        while True:
            if cur_set_i >= num_files:
                cur_set_i = 0
                random.shuffle(files)

            self.train_spectra_loader.load_spectra([files[cur_set_i]], del_old=True)
            cur_set_i += 1
            #spectra.append(self.transform_train(encoded=encoded))
            dat = self.transform_train(encoded=encoded)
            #spectra_x = np.concatenate((spectra_x, dat[0]))
            #spectra_y = np.concatenate((spectra_y, dat[1]))
            #print("Append start")
            #spectra_x.extend(dat[0].tolist())
            #spectra_y.extend(dat[1].tolist())

            if spectra_x is None:
                spectra_x = dat[0]
            else:
                spectra_x = np.concatenate((spectra_x, dat[0]))

            if spectra_y is None:
                spectra_y = dat[1]
            else:
                spectra_y = np.concatenate((spectra_y, dat[1]))
            #print("Append stop")


            #print(spectra_x)
            #print(len(spectra_x))

            while len(spectra_x) >= batch_size:
                spectra_batch_x = spectra_x[:batch_size]
                spectra_batch_y = spectra_y[:batch_size]
                spectra_x = spectra_x[batch_size:]
                spectra_y = spectra_y[batch_size:]

                yield spectra_batch_x, spectra_batch_y

    def get_num_training_files(self):
        return len(self.train_spectra_loader.get_data_files())


class BaseModel(ABC):
    """ Abstract class for our models to extend. """

    def __init__(self, keras_model):
        self.keras_model = keras_model
        self.test_results = None
        self.compile_dict = None
        self.batch_size = None
        self.epochs = None
        self.validation_size = None
        self.history = None

    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs, compile_dict, validation_size=0.20):
        self.compile(compile_dict)

        self.keras_model.fit(X_train, y_train, validation_split=validation_size, epochs=epochs, batch_size=batch_size)
        self.compile_dict = compile_dict
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_size = validation_size
        self.evaluate(X_test, y_test)
        self.history = self.get_model_history()

    def fit_generator(self, preprocessor, train_size, X_test, y_test, batch_size, epochs, compile_dict, validation_size=0.20, encoded=False):
        self.compile(compile_dict)
        self.keras_model.fit_generator(preprocessor.train_generator(batch_size=batch_size, encoded=encoded),
                                       steps_per_epoch=train_size//batch_size, validation_data=(X_test, y_test),
                                       epochs=epochs)
        self.compile_dict = compile_dict
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_size = validation_size
        self.evaluate(X_test, y_test)
        self.history = self.get_model_history()

    def compile(self, compile_dict):
        self.keras_model.compile(**compile_dict)

    def get_model_config(self):
        return self.keras_model.to_json()

    def get_model_history(self):
        return self.keras_model.history.history

    def evaluate(self, X_test, y_test):
        self.test_results = self.keras_model.evaluate(X_test, y_test)

    def to_dict(self):
        params = dict()
        params['config'] = self.get_model_config()
        params['compile_dict'] = self.compile_dict
        params['batch_size'] = self.batch_size
        params['epochs'] = self.epochs
        params['history'] = self.get_model_history()
        params['test_results'] = self.test_results
        params['weights'] = self.keras_model.get_weights()
        return params

