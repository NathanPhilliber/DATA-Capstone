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

