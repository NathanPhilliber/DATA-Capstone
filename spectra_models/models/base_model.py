from abc import ABC
from abc import abstractmethod
from spectra_generator import Spectrum, SpectraLoader, SpectraGenerator
from sklearn.preprocessing import OneHotEncoder
import numpy as np

MODEL_RESULTS_PATH = 'spectra_models/model_results.txt'


def get_model_params(model_name, train_path, test_path, model):
    model_dict = dict()
    model_dict['name'] = model_name
    model_dict['train_path'] = train_path
    model_dict['test_path'] = test_path
    model_dict['model_params'] = model.to_dict()
    return model_dict


def save_model(model_name, train_path, test_path, model):
    model_configs = get_model_params(model_name, train_path, test_path, model)
    with open(MODEL_RESULTS_PATH, 'a', encoding='UTF-8') as f:
        f.write(str(model_configs) + '\n')
    print(f'Saved model results as {model_name} in the file: {MODEL_RESULTS_PATH}')


class SpectraPreprocessor:
    def __init__(self, train_filename, test_filename):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.train_spectra_loader = SpectraLoader(self.train_filename)
        self.test_spectra_loader = SpectraLoader(self.test_filename)
        self.one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')

    def get_data(self, loader):
        dm = np.array(loader.get_dm())
        X = dm.reshape(dm.shape[0], dm.shape[2], dm.shape[1], 1)
        del dm
        y = np.array(loader.get_n())
        y = y.reshape(y.shape[0], 1)
        return X, y

    def transform(self):
        X_train, y_train = self.get_data(self.train_spectra_loader)
        X_test, y_test = self.get_data(self.test_spectra_loader)
        y_train_enc = self.one_hot_encoder.fit_transform(y_train)
        y_test_enc = self.one_hot_encoder.transform(y_test)
        del y_train, y_test
        return X_train, y_train_enc, X_test, y_test_enc


class BaseModel(ABC):
    """ Abstract class for our models to extend. """

    def __init__(self, keras_model):
        self.keras_model = keras_model
        self.y_test = None
        self.y_test_pred = None
        self.compile_dict = None
        self.batch_size = None
        self.epochs = None
        self.validation_size = None

    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs, compile_dict, validation_size=0.20):
        self.compile(compile_dict)
        self.keras_model.fit(X_train, y_train, validation_split=validation_size, epochs=epochs, batch_size=batch_size)
        self.compile_dict = compile_dict
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_size = validation_size
        self.evaluate(X_test, y_test)

    def compile(self, compile_dict):
        self.keras_model.compile(**compile_dict)

    def get_model_config(self):
        return self.keras_model.to_json()

    def get_model_history(self):
        return self.keras_model.history.history

    def evaluate(self, X_test, y_test):
        self.y_test = y_test
        self.y_test_pred = self.keras_model.predict(X_test)

    def to_dict(self):
        params = dict()
        params['config'] = self.get_model_config()
        params['compile_dict'] = self.compile_dict
        params['batch_size'] = self.batch_size
        params['epochs'] = self.epochs
        params['history'] = self.get_model_history()
        params['evaluate'] = {'y_test': self.y_test, 'y_test_pred': self.y_test_pred}
        return params

