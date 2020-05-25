
# Spectra Generator: Dataset Generation

## Overview Description
This directory contains code used to generate spectra based on parameter which the user specifies. The data generated in this module is later used to train models.


## File Structure
```bash
├── datagen     <--------------------------  root directory for data-generating code
│   ├── run_gen.py      <-------------------- driver program (execute with python3)
│   ├── matlab_scripts     <-------------------- stores MATLAB scripts used to generate data
│   │   ├── spectra_generator_v1.m
│   │   ├── spectra_generator_v2.m
│   ├── spectra_loader.py    <-----------------------  load spectra data that has already been generated
│   ├── spectrum.py    <---------  load data from a single spectrum that has already been generated
│   ├── loadmatlab.py    <---------------- load spectrum from MATLAB scripts 
│   └── notebooks   <------------------  Jupyter notebooks for exploration
│   └── spectra_generator.py     <------------------  Use MATLAB scripts to generate spectra data.
│   └── reshard.py     <------------------  Load data that has been split into numerous shards
```

## Installation Instructions:
We require the installation of MATLAB in order to be able to generate data. If you don't wish to install MATLAB and are only interested in testing out a model without generating additional data, we have placed an example dataset in the following directory: [example_set](../data/datasets/example_set). However, if you wish to generate additional datasets, make sure to complete the following steps:

*MAC OS*
1. Find your 'matlabroot'
  - Open Matlab Application
  - In command window, enter '`matlabroot`'
  - Note result (refer to as '`MATLABROOT`')
2. Open terminal (either in Jupyter or machine)
  - `cd MATLABROOT/extern/engines/python`
  - `python setup.py install`

## Running Code
In order to execute these scripts, you will need to have `python3` and all the [dependencies](../requirements.txt) installed.

The executable is named `run_gen.py`. An example execution of it may look like the following:
```bash
python3 -m datagen.run_gen
```

- Specify the name of the directory to store the data in.
```bash
Spectra are stored in this directory --  data/datasets/:
```

- Select the MATLAB script used to generate the data.
```bash
Select from the following MATLAB scripts, located in: datagen/matlab_scripts
0: spectra_generator_v1.m
1: spectra_generator_v2.m
```

- Specify the number of instances to create.
```bash
Number of instances to create [10000]:
```

- Specify the number of spectra to put in each shard.
```bash
How many spectra to put in each shard (0 = no shard) [0]:
```

- Specify the number of channels that each spectrum will have.
```bash
Number of channels to generate [10.0]:
```

- Specify the maximum number of modes or resonances.
```bash
Maximum number of modes [5.0]:
```

- Specify the maximum number of shell modes.
```bash
Maximum number of shell peaks [5.0]:
```

- Specify the scale.
```bash
Scale or width of window [1.0]:
```

- Specify omega shift. (This will change the resolution of the x-axis)
```bash
Omega Shift [10.0]:
```

- Specify the variation of gamma (or dG).
```bash
Variation of Gamma [0.5]:
```

- Specify the variation of gamma for the shell modes (or dGs).
```bash
Gamma variation of shell modes [1.8]:
```

------------------

- After specifying the options above, you should receive a similar message:
```bash
Generating 500 spectra for shard #1 (500 left)...
  Making SpectraLoader...
  Splitting data...
    425 Train, 75 Test
  Saving training data...
    Saved 425 spectra
  Saving testing data...
    Saved 75 spectra
```

- Suppose you decide to name your dataset: `example_set`.
```
├── datagen       
│   ├── datasets  
│   │   ├── example_set
│   │   │   ├── gen_info.json
│   │   │   ├── train_example_set.pkl
│   │   │   ├── test_example_set.pkl
```

- The `gen_info.json` file will store the configurations used when generating this dataset:
```json
{
    "n_max": 4.0,
    "n_max_s": 5.0,
    "num_channels": 50.0,
    "scale": 1.0,
    "omega_shift": 10.0,
    "dg": 0.5,
    "dgs": 0.5,
    "num_timesteps": 301,
    "num_instances": 500,
    "matlab_script": "spectra_generator_v2.m"
}
```