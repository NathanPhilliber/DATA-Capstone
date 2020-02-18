from utils import *
from abc import ABC
from abc import abstractmethod
import json
from datetime import datetime


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

    def save(self, class_name, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(MODEL_RES_DIR, class_name + "." + str(datetime.now().strftime("%m%d.%H%M")))

        try_create_directory(save_dir)

        weights_path = os.path.join(save_dir, WEIGHTS_FILENAME)
        info_path = os.path.join(save_dir, TRAIN_INFO_FILENAME)
        self.keras_model.save_weights(weights_path)

        info_dict = self.get_info_dict()
        info_dict["class_name"] = class_name
        json.dump(info_dict, open(info_path, "w"))

        return save_dir

    def persist(self, dirname, result_dir=MODEL_RES_DIR):
        model_directory = os.path.join(result_dir, dirname)
        weights_path = os.path.join(model_directory, WEIGHTS_FILENAME)
        info_path = os.path.join(model_directory, TRAIN_INFO_FILENAME)
        info = json.load(open(info_path, 'r'))

        self.compile_dict = info['compile_dict']
        self.batch_size = info['batch_size']
        self.epochs = info['epochs']
        self.history = info['history']
        self.test_results = info['test_results']

        self.compile(self.compile_dict)
        self.keras_model.load_weights(weights_path)