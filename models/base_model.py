from utils import *
from abc import ABC
from abc import abstractmethod
import json


class BaseModel(ABC):
    """ Abstract class for our networks to extend. """

    @abstractmethod
    def build_model(self, num_channels, num_timesteps, output_shape):
        pass

    def __init__(self, num_channels, num_timesteps, output_shape):
        self.keras_model = self.build_model(int(num_channels), int(num_timesteps), int(output_shape))
        self.test_results = None
        self.compile_dict = None
        self.batch_size = None
        self.epochs = 0
        self.validation_size = None
        self.history = None

    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs, compile_dict, validation_size=0.20):
        self.compile(compile_dict)

        self.keras_model.fit(X_train, y_train, validation_split=validation_size, epochs=epochs, batch_size=batch_size)
        self.compile_dict = compile_dict
        self.batch_size = batch_size
        self.epochs += epochs
        self.validation_size = validation_size
        self.evaluate(X_test, y_test)
        self.history = self.get_model_history()

    def fit_generator(self, preprocessor, train_size, X_test, y_test, batch_size, epochs, compile_dict,
                      validation_size=0.20, encoded=False):
        self.compile(compile_dict)
        self.keras_model.fit_generator(preprocessor.train_generator(batch_size=batch_size, encoded=encoded),
                                       steps_per_epoch=train_size//batch_size, validation_data=(X_test, y_test),
                                       epochs=epochs)
        self.compile_dict = compile_dict
        self.batch_size = batch_size
        self.epochs += epochs
        self.validation_size = validation_size
        self.evaluate(X_test, y_test)
        self.history = self.get_model_history()

    def compile(self, compile_dict):
        self.keras_model.compile(**compile_dict)

    def get_model_config(self):
        return self.keras_model.to_json()

    def get_model_history(self):
        history = self.keras_model.history.history
        for key, value in history.items():
            history[key] = [float(v) for v in value]
        return history

    def evaluate(self, X_test, y_test):
        self.test_results = self.keras_model.evaluate(X_test, y_test)

    def get_info_dict(self):
        params = dict()
        params['compile_dict'] = self.compile_dict
        params['batch_size'] = self.batch_size
        params['epochs'] = self.epochs
        params['history'] = self.get_model_history()
        params['test_results'] = self.test_results
        return params

    def save(self, class_name, save_path='models/persist'):
        model_directory = save_path + '/' + class_name
        try_create_directory(model_directory)
        weights_path = model_directory + '/weights.h5'
        info_path = model_directory + '/info.json'
        self.keras_model.save_weights(weights_path)

        info_dict = self.get_info_dict()
        json.dump(info_dict, open(info_path, "w"))

    def persist(self, class_name, save_path='models/persist'):
        model_directory = save_path + '/' + class_name
        weights_path = model_directory + '/weights.h5'
        info_path = model_directory + '/info.json'
        info = json.load(open(info_path, 'r'))

        self.compile_dict = info['compile_dict']
        self.batch_size = info['batch_size']
        self.epochs = info['epochs']
        self.history = info['history']
        self.test_results = info['test_results']

        self.compile(self.compile_dict)
        self.keras_model.load_weights(weights_path)








