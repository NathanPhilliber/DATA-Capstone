from models.networks.abstract_models.ensemble_model import EnsembleModel
from models.networks.abstract_models.base_model import BaseModel

from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense, concatenate

import numpy as np


class CNNEnsemble1(BaseModel):
    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs, compile_dict=None, validation_size=0.20):
        X_train_reshape = np.moveaxis(X_train, 2, 0)[..., np.newaxis].tolist()
        X_test_reshape = np.moveaxis(X_test, 2, 0)[..., np.newaxis].tolist()
        super().fit(X_train_reshape, y_train, X_test_reshape, y_test, batch_size, epochs, compile_dict, validation_size)

    def fit_generator(self, preprocessor, train_size, X_test, y_test, batch_size, epochs, compile_dict=None,
                      validation_size=0.20, encoded=False):
        X_test_reshape = np.moveaxis(X_test, 2, 0)[..., np.newaxis].tolist()
        super().fit_generator(preprocessor, train_size, X_test_reshape, y_test, batch_size, epochs, compile_dict,
                              validation_size, encoded)

    def set_params_range(self):
        return {'conv1d_filters_1': {'type': 'integer', 'min': 10, 'max': 500, 'default': 128},
                'conv1d_filters_2': {'type': 'integer', 'min': 5, 'max': 250, 'default': 64},
                'conv1d_kernel_1': {'type': 'integer', 'min': 1, 'max': 10, 'default': 5},
                'conv1d_kernel_2': {'type': 'integer', 'min': 1, 'max': 10, 'default': 5},
                'dropout_1': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5},
                'dense_size_1': {'type': 'integer', 'min': 5, 'max': 250, 'default': 64},
                'dropout_2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5},
                'dense_size_2': {'type': 'integer', 'min': 5, 'max': 125, 'default': 32},
                'dense_size_3': {'type': 'integer', 'min': 5, 'max': 125, 'default': 32}}

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        model = EnsembleModel.builder() \
            .with_input_channels(num_channels) \
            .with_homogeneous_models(True) \
            .build()

        print(params['dense_size_3'] * num_channels)

        model.add_input_layers(input_shape=(num_timesteps, 1))
        model.add_layer(Conv1D(params['conv1d_filters_1'], params['conv1d_kernel_1'], activation='relu'))
        model.add_layer(Conv1D(params['conv1d_filters_2'], params['conv1d_kernel_2'], activation='relu'))
        model.add_layer(Flatten())
        model.add_layer(Dropout(rate=params['dropout_1']))
        model.add_layer(Dense(params['dense_size_1']))
        model.add_layer(Dropout(rate=params['dropout_2']))
        model.add_layer(Dense(params['dense_size_2']))
        model.merge_sub_models(func=concatenate)
        model.add_layer(Dense(params['dense_size_3'] * num_channels))
        model.add_layer(Dense(output_shape))
        model.compile()
        return model.keras_model
