from models.networks.abstract_models.ensemble_model import EnsembleModel
from models.networks.abstract_models.base_model import BaseModel

from keras.layers import Conv1D, Flatten, Dropout, Dense, concatenate


class CNNEnsemble1(BaseModel):
    def build_model(self, num_channels, num_timesteps, output_shape):
        model = EnsembleModel.builder() \
            .with_input_channels(num_channels) \
            .with_homogeneous_models(True) \
            .build()

        model.add_input_layers(input_shape=(num_timesteps, 1))
        model.add_layer(Conv1D(128, 5, activation='relu'))
        model.add_layer(Conv1D(64, 5, activation='relu'))
        model.add_layer(Flatten())
        model.add_layer(Dropout(rate=0.5))
        model.add_layer(Dense(64))
        model.add_layer(Dropout(rate=0.5))
        model.add_layer(Dense(32))
        model.merge_sub_models(func=concatenate)
        model.add_layer(Dense(32 * num_channels))
        model.add_layer(Dense(output_shape))
        model.compile()
        return model.keras_model
