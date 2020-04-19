from utils import *
from comet_ml import Experiment, Optimizer, ExistingExperiment
from abc import ABC
from abc import abstractmethod
import json
from datetime import datetime
from sklearn.metrics import classification_report
import numpy as np
import pickle


class BaseModel(ABC):
    """ Abstract class for our networks to extend. """

    @abstractmethod
    def set_params_range(self):
        pass

    @abstractmethod
    def build_model(self, num_channels, num_timesteps, output_shape, params):
        pass

    def __init__(self, num_channels, num_timesteps, output_shape, use_comet=True):
        self.params_range = self.set_params_range()
        self.params = None
        self.keras_model = None
        self.num_channels = int(num_channels)
        self.num_timesteps = int(num_timesteps)
        self.output_shape = int(output_shape)
        self.test_results = None
        self.compile_dict = None
        self.batch_size = None
        self.epochs = 0
        self.validation_size = None
        self.history = None
        self.preds = None
        self.weights_path = None

    def get_default_params(self):
        return {k: v['default'] for k, v in self.params_range.items()}

    def _fit_preinit(self, compile_dict):
        if self.params is None:
            self.params = self.get_default_params()
            print(f"Using default parameters: {self.params}")

        self.keras_model = self.build_model(self.num_channels, self.num_timesteps, self.output_shape, self.params)
        if self.weights_path is not None:
            self.keras_model.load_weights(self.weights_path)

        if compile_dict is not None:
            self.compile(compile_dict)
            self.compile_dict = compile_dict
        elif self.compile_dict is not None:
            self.compile(self.compile_dict)

    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs, compile_dict=None, validation_size=0.20):
        self._fit_preinit(compile_dict)

        self.keras_model.fit(X_train, y_train, validation_split=validation_size, epochs=epochs, batch_size=batch_size)
        self._fit_complete(X_test, y_test, batch_size, epochs, validation_size)

    def fit_generator(self, preprocessor, train_size, X_test, y_test, batch_size, epochs, compile_dict=None,
                      validation_size=0.20, encoded=False):

        self._fit_preinit(compile_dict)

        self.keras_model.fit_generator(preprocessor.train_generator(batch_size=batch_size, encoded=encoded),
                                       steps_per_epoch=train_size//batch_size, validation_data=(X_test, y_test),
                                       epochs=epochs)
        self._fit_complete(X_test, y_test, batch_size, epochs, validation_size)

    def _fit_complete(self, X_test, y_test, batch_size, epochs, validation_size=0.20):
        self.batch_size = batch_size
        self.epochs += epochs
        self.validation_size = validation_size
        self.test_results = self.evaluate(X_test, y_test)
        self.history = BaseModel._merge_histories(self.history, self.get_model_history())
        self.preds = self.get_preds(X_test, y_test)
        #self.get_classification_report(*self.preds)

    def compile(self, compile_dict):
        self.keras_model.compile(**compile_dict)

    def get_model_config(self):
        return self.keras_model.to_json()

    def get_model_history(self):
        history = self.keras_model.history.history
        for key, value in history.items():
            history[key] = [float(round(v, 5)) for v in value]
        return history

    def evaluate(self, X_test, y_test):
        eval_res = self.keras_model.evaluate(X_test, y_test)
        results = {"metrics_names": self.keras_model.metrics_names,
                   "metrics": [float(val) for val in eval_res]}
        return results

    def get_preds(self, X_test, y_test):
        preds = self.keras_model.predict(X_test)
        return y_test, preds

    def serialize(self):
        params = dict()
        params['compile_dict'] = self.compile_dict
        params['batch_size'] = self.batch_size
        params['epochs'] = self.epochs
        params['history'] = BaseModel._merge_histories(self.history, self.get_model_history())
        params['test_results'] = self.test_results

        return params

    def save(self, class_name, dataset_name, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(MODEL_RES_DIR, class_name + RESULT_DIR_DELIM + dataset_name + "." + str(datetime.now().strftime("%m%d.%H%M")))

        try_create_directory(save_dir)
        weights_path = os.path.join(save_dir, WEIGHTS_FILENAME)
        info_path = os.path.join(save_dir, TRAIN_INFO_FILENAME)
        self.keras_model.save_weights(weights_path)

        info_dict = self.serialize()
        info_dict["class_name"] = class_name
        info_dict["dataset_name"] = dataset_name
        json.dump(info_dict, open(info_path, "w"))

        return save_dir

    def persist(self, dirname, result_dir=MODEL_RES_DIR):
        model_directory = os.path.join(result_dir, dirname)

        self.weights_path = os.path.join(model_directory, WEIGHTS_FILENAME)

        info_path = os.path.join(model_directory, TRAIN_INFO_FILENAME)
        info = json.load(open(info_path, 'r'))

        self.compile_dict = info['compile_dict']
        self.batch_size = info['batch_size']
        self.epochs = info['epochs']
        self.history = info['history']
        self.test_results = info['test_results']

    @staticmethod
    def _merge_histories(hist1, hist2):
        if hist1 is None and hist2 is None:
            return {}
        elif hist1 is None:
            return hist2
        elif hist2 is None:
            return hist1

        assert hist1.keys() == hist2.keys(), "incompatible histories to merge"

        hist = {}
        for key in hist1:
            hist[key] = hist1[key] + hist2[key]

        return hist




