# Neural Networks: Architecture, Training, and Evaluation

## Overview Description
This directory contains code used for

 - defining neural network model architectures
 - training models with spectral data ([generated via `datagen/`](../datagen)) 
 - serializing and resuming training sessions
 - evaluating the accuracy of a trained model
 - generating graphs and tables to visualize trained model results

## File Structure

```bash
├── models     <--------------------------  root directory for network related code
│   ├── run_train.py     <--------------------  driver program (execute with python3)
│   ├── networks/     <-----------------------  network architectures
│   │   ├── abstract_models/     <----------------  code common to amongst models
│   │   ├── ensemble_models.py     <--------------  ensemble models architecture file
│   │   └── lstm_models.py     <------------------  LSTM models architecture file
│   ├── spectra_preprocessor.py     <---------  code to read data and transform it into train-ready matrices
│   ├── evaluator.py     <----------------  trained model evaluation code
│   └── notebooks/     <------------------  directory containing "scratch work" code and experiments
```

## Running Code

In order to execute these scripts, you will need to have `python3` and all the [dependencies](../requirements.txt) installed.

The executable is named `run_train.py`. An example execution of it may look like the following:
```bash
python3 run_train.py < new | continue | evaluate >
```

If you are trying to run the script with a GPU, refer to the 'Docker and GPU' instructions.

The script must be run with one of the three options above. These options are detailed below:

### Training a New Model
If you want to train a brand new model, then this is the option you will want to select: `new`

To view the arguments for this option, run: `python3 run_train.py new --help`

If you do not specify arguments, they will be prompted one-by-one in the terminal.

Requirements:
 - At least one dataset in `data/datasets/` (or wherever the data path is as specified in `utils.py`)
 - At least one model architecture class in `models/networks/`. The script will automatically detect any class 
   that extends `models/networks/abstract_models/base_model.py:BaseModel` and is located in the networks directory as 
   a valid model architecture. You may define more than one model architecture per file.

### Training an Existing Model
If you want to continue training a model that has been previously trained, then you will select this option: `continue`

To view the arguments for this option, run: `python3 run_train.py continue --help`

If you do not specify arguments, they will be prompted one-by-one in the terminal.
Note: one of the arguments will be prompted via terminal and is not available as a command line argument.

Requirements:
 - At least one dataset in `data/datasets/` (or wherever the data path is as specified in `utils.py`). You may use a 
   different dataset than was previously used, however this has not been extensively tested.
 - At least one model architecture class in `models/networks/`. The script will automatically detect any class 
   that extends `models/networks/abstract_models/base_model.py:BaseModel` and is located in the networks directory as 
   a valid model architecture. You may define more than one model architecture per file.
 - At least one training session in `data/results/` (or wherever the results path is as specified in `utils.py`). 
   The training session must match the model architecture.

### Evaluating a Model
If you want to use a previously trained model to predict from a test set, then you will select this option: `evaluate`

To view the arguments for this option, run: `python3 run_train.py evaluate --help`

If you do not specify arguments, they will be prompted one-by-one in the terminal.
Note: one of the arguments will be prompted via terminal and is not available as a command line argument.

Requirements:
 - At least one dataset in `data/datasets/` (or wherever the data path is as specified in `utils.py`). You may use a 
   different dataset than was previously used, however this has not been extensively tested.
 - At least one model architecture class in `models/networks/`. The script will automatically detect any class 
   that extends `models/networks/abstract_models/base_model.py:BaseModel` and is located in the networks directory as 
   a valid model architecture. You may define more than one model architecture per file.
 - At least one training session in `data/results/` (or wherever the results path is as specified in `utils.py`). 
   The training session must match the model architecture.
   
Note: only the test portion of the dataset will be used.

## Defining Neural Network Architectures

Creating new architecture is easy. There are only two requirements:
1. Extend `BaseModel` located in `models/networks/abstract_models/base_model.py`
2. Exist in a file in `models/networks/`. The execute `run_train.py` script will automatically detect architectures
   in this location.
   
Here is a full example of a network architecture:
```python
class GoogleModel(BaseModel):

    def set_params_range(self):
        return {'conv_1':  {'type': 'integer', 'min': 8, 'max': 64,  'default': 16},
                'conv_2':  {'type': 'integer', 'min': 8, 'max': 64,  'default': 32},
                'bi_1':    {'type': 'float',   'min': 8, 'max': 128, 'default': 128},
                'bi_2':    {'type': 'integer', 'min': 8, 'max': 128, 'default': 128},
                'drop_1':  {'type': 'float',   'min': 0, 'max': 1,   'default': 0.01},
                'dense_1': {'type': 'float',   'min': 8, 'max': 128, 'default': 64},
                'drop_2':  {'type': 'float',   'min': 0, 'max': 1,   'default': 0.05}
                }

    def build_model(self, num_channels, num_timesteps, output_shape, params):
        num_attention = num_timesteps - 8
        model = Sequential()
        model.add(Conv1D(params['conv_1'], 5, input_shape=(num_timesteps, num_channels)))
        model.add(BatchNormalization())
        model.add(Conv1D(params['conv_2'], 5))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(params['bi_1'], return_sequences=True)))
        model.add(Bidirectional(LSTM(params['bi_2'], return_sequences=True)))
        model.add(Attention(num_attention))
        model.add(Dropout(params['drop_1']))
        model.add(Dense(params['dense_1'], activation='elu'))
        model.add(Dropout(params['drop_2']))
        model.add(Dense(output_shape, activation='softmax'))
        return model
```

Architecture classes must include `build_model` and `set_params_range` functions.
