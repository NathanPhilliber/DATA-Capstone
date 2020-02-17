from models.networks.ensemble_models import EnsembleModel
from keras.layers import Conv1D, Flatten, Dropout, Dense, concatenate


LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'
INPUT_CHANNELS = 10
METRICS = ['accuracy']

model = EnsembleModel.builder()\
    .with_input_channels(INPUT_CHANNELS)\
    .with_homogeneous_models(True)\
    .build()

model.add_input_layers(input_shape=(1000, 1))
model.add_layer(Conv1D(128, 5, activation='relu'))
model.add_layer(Conv1D(64, 5, activation='relu'))
model.add_layer(Flatten())
model.add_layer(Dropout(rate=0.5))
model.add_layer(Dense(64))
model.add_layer(Dropout(rate=0.5))
model.add_layer(Dense(32))
model.merge_sub_models(func=concatenate)
model.add_layer(Dense(32 * INPUT_CHANNELS))
model.add_layer(Dense(1))
model.compiler()\
    .with_loss(LOSS)\
    .with_metrics(METRICS)\
    .with_optimizer(OPTIMIZER)\
    .compile()

print(model.summary())
