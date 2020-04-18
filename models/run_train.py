from utils import *
import os

import json
from comet_ml import Optimizer
import numpy as np
from sklearn.metrics import classification_report
from models.spectra_preprocessor import SpectraPreprocessor
from datagen.spectra_loader import SpectraLoader
from datetime import datetime
import click
import tensorflow as tf


COMPILE_DICT = {'optimizer': 'adam','loss': 'categorical_crossentropy', 'metrics': ['accuracy', 'mae', 'mse']}
OPTIMIZE_PARAMS = {'algorithm': 'bayes', 'spec': {'metric': 'loss', 'objective': 'minimize'}}

GENERATOR_LIMIT = 10000  # The minimum number of data points where fit generator should be used


loaded_models = None
def get_loaded_models():
    global loaded_models
    if loaded_models is None:
        loaded_models = get_modules(NETWORKS_DIR)

    return loaded_models


@click.group()
def main():
    pass


def get_module(model_module_index):
    module, package_name = get_loaded_models()[model_module_index]

    return module, package_name


def load_model(module, class_name, num_channels, n_max, num_timesteps):
    model_class = getattr(module, class_name)
    model = model_class(num_channels, num_timesteps, n_max)
    return model


def load_dataset_info(dataset_name):
    dataset_config = json.load(open(os.path.join(DATA_DIR, dataset_name, DATAGEN_CONFIG), "r"))
    return dataset_config


def load_data(dataset_name, dataset_config):
    use_generator = dataset_config["num_instances"] > GENERATOR_LIMIT
    spectra_pp = SpectraPreprocessor(dataset_name=dataset_name, use_generator=use_generator)
    return spectra_pp


def get_prior_config(result_dirname):
    result_path = os.path.join(MODEL_RES_DIR, result_dirname)
    result_info = json.load(open(os.path.join(result_path, TRAIN_INFO_FILENAME), "rb"))
    return result_info


def initialize_model(dataset_name, model_name, model_module_index):
    dataset_config = load_dataset_info(dataset_name)
    module, package_name = get_module(model_module_index)
    model = load_model(module, model_name, dataset_config['num_channels'], dataset_config['n_max'], dataset_config['num_timesteps'])
    return dataset_config, model


def train_model(model, dataset_name, dataset_config, batch_size, n_epochs, compile_dict=None):
    use_generator = dataset_config["num_instances"] > GENERATOR_LIMIT
    spectra_pp = SpectraPreprocessor(dataset_name=dataset_name, use_generator=use_generator)
    #model.log_imgs(dataset_name)
    #model.log_script(dataset_config)

    if use_generator:
        print("\nUsing fit generator.\n")
        X_test, y_test = spectra_pp.transform_test(encoded=True)
        model.fit_generator(spectra_pp, spectra_pp.datagen_config["num_instances"], X_test,
                            y_test, batch_size=batch_size, epochs=n_epochs, encoded=True,
                            compile_dict=compile_dict)

    else:
        X_train, y_train, X_test, y_test = spectra_pp.transform(encoded=True)
        model.fit(X_train, y_train, X_test, y_test, batch_size=batch_size, epochs=n_epochs,
                  compile_dict=compile_dict)

    return model


def prompt_dataset_string():
    data_dirs = sorted(os.listdir(DATA_DIR))
    data_dirs = [data_dir for data_dir in data_dirs if os.path.isdir(os.path.join(DATA_DIR, data_dir))]

    msg = ""

    msg += f"\nThe following datasets were found in {to_local_path(DATA_DIR)}:\n"
    msg += f"{'Selection':10} {'Set Name':15} {'Num Spectra':15} {'Num Channels':15}\n"
    for dir_i, dir_name in enumerate(data_dirs):
        config = SpectraLoader.read_dataset_config(dir_name)
        msg += f"  {dir_i:6}:  {dir_name:15} {format(config['num_instances'], ','):15} {int(config['num_channels']):2}\n"

    msg += "\nSelect dataset to use"

    return msg


def get_dataset_name(ctx, param, dataset_name_or_selection):
    data_dirs = sorted(os.listdir(DATA_DIR))
    data_dirs = [data_dir for data_dir in data_dirs if os.path.isdir(os.path.join(DATA_DIR, data_dir))]

    try:
        selection = int(dataset_name_or_selection)
        if selection >= len(data_dirs) or selection < 0:
            raise Exception("Invalid option: %d out of range" % selection)

        dataset_name = data_dirs[selection]
    except ValueError:
        dataset_name = dataset_name_or_selection

    if dataset_name not in data_dirs:
        raise Exception("Could not find dataset with set_name='%s' in '%s" % (dataset_name, DATA_DIR))

    ctx.params["dataset_name"] = dataset_name
    return dataset_name


def prompt_model_string():
    list_i = 0
    names = []
    msg = ""

    for module_i, (module, module_name) in enumerate(get_loaded_models()):
        classes = sorted(get_classes(module, module_name))

        for class_i, class_name in enumerate(classes):
            list_i += 1
            names.append(class_name)

    msg += f"\nThe following models were found in {to_local_path(NETWORKS_DIR)}:\n"
    for sel_i, class_name in enumerate(names):
        msg += f"  {sel_i}:\t {class_name}\n"

    msg += "\nSelect model to run"

    return msg


def get_model_name(ctx, param, model_name_or_selection):
    list_i = 0
    names = []
    module_indices = []
    loaded_models = get_loaded_models()

    for module_i, (module, module_name) in enumerate(loaded_models):
        classes = sorted(get_classes(module, module_name))

        for class_i, class_name in enumerate(classes):
            list_i += 1
            names.append(class_name)
            module_indices.append(module_i)

    try:
        selection = int(model_name_or_selection)
        if selection >= len(names) or selection < 0:
            raise Exception("Invalid option: %d out of range (0, %d)" % (selection, len(names)))

        model_name = names[selection]
    except ValueError:
        model_name = model_name_or_selection

    if model_name not in names:
        raise Exception("Could not find model with model_name='%s' in '%s'" % (model_name, NETWORKS_DIR))

    ctx.params["model_name"] = model_name
    ctx.params["model_module_index"] = module_indices[names.index(model_name)]
    return model_name


def prompt_previous_run(model_name):
    result_dirs = os.listdir(MODEL_RES_DIR)
    result_dirs = sorted([result_dir for result_dir in result_dirs if model_name == os.path.basename(result_dir).split("_")[0]])

    msg = ""
    msg += "Found Existing Runs:\n"
    for dir_i, result_dir in enumerate(result_dirs):
        msg += f"  {dir_i:3}:  {result_dir}\n"

    msg += "\nSelect model to train"
    return msg


def get_result_name(model_name, result_name_or_selection):
    result_dirs = os.listdir(MODEL_RES_DIR)
    result_dirs = sorted([result_dir for result_dir in result_dirs if model_name == os.path.basename(result_dir).split("_")[0]])

    try:
        selection = int(result_name_or_selection)
        if selection >= len(result_dirs) or selection < 0:
            raise Exception("Invalid option: %d out of range (0, %d)" % (selection, len(result_dirs)))

        result_name = result_dirs[selection]
    except ValueError:
        result_name = result_name_or_selection

    return result_name


@main.command(name="continue", help="Continue training an existing run")
@click.option('--model-name', "-m", prompt=prompt_model_string(), callback=get_model_name, default=None)
@click.option('--dataset-name', "-d", prompt=prompt_dataset_string(), callback=get_dataset_name, default=None)
@click.option("--n-epochs", prompt="Number of epochs", default=DEFAULT_N_EPOCHS, type=click.IntRange(min=1))
def continue_train_model(model_name, dataset_name, n_epochs, model_module_index=None):
    result_name = get_result_name(model_name, input(prompt_previous_run(model_name)))  #  If you can figure out how to add this to Click args, then please do

    dataset_config, model = initialize_model(dataset_name, model_name, model_module_index)
    #result_info = get_prior_config(model_name)

    model.persist(result_name)
    model = train_model(model, dataset_name, dataset_config, model.batch_size, n_epochs)

    #y_true, y_pred = model.preds
    #labels = [str(i) for i in range(1, int(dataset_config['n_max'] + 1))]
    #model.experiment.log_confusion_matrix(y_true, y_pred, labels=labels)
    #y_true = np.argmax(y_true, axis=1)
    #y_pred = np.argmax(y_pred, axis=1)

    #print(classification_report(y_true, y_pred, target_names=[1, 2, 3, 4]))

    save_loc = model.save(model_name, dataset_name)
    print(f"Saved model to {to_local_path(save_loc)}")


@main.command(name="new", help="Train a new model")
@click.option('--model-name', "-m", prompt=prompt_model_string(), callback=get_model_name, default=None)
@click.option('--dataset-name', "-d", prompt=prompt_dataset_string(), callback=get_dataset_name, default=None)
@click.option("--batch-size", "-bs", prompt="Batch size", default=DEFAULT_BATCH_SIZE, type=click.IntRange(min=1))
@click.option("--n-epochs", "-n", prompt="Number of epochs", default=DEFAULT_N_EPOCHS, type=click.IntRange(min=1))
@click.option('--use-comet/--no-comet', is_flag=True, default=True)
@click.option("--comet-name", "-cn", prompt="What would you like to call this run on comet?", default=f"model-{str(datetime.now().strftime('%m%d.%H%M'))}")
def train_new_model(comet_name, batch_size, n_epochs, dataset_name, model_name, use_comet, model_module_index=None):
    print("Using dataset:", dataset_name)
    print("Using model:", model_name)

    dataset_config, model = initialize_model(dataset_name, model_name, model_module_index)

    #if use_comet:
        #model.load_comet_new(comet_name, dataset_config)

    model = train_model(model, dataset_name, dataset_config, batch_size, n_epochs, compile_dict=COMPILE_DICT)

    #if model.experiment is not None:
    #    y_true, y_pred = model.preds
    #    model.experiment.log_parameters(model.serialize())
    #    model.experiment.log_confusion_matrix(y_true, y_pred)

    #    labels = [str(i) for i in range(1, int(dataset_config['n_max'] + 1))]
    #    model.experiment.log_confusion_matrix(y_true, y_pred, labels=labels)

    save_loc = model.save(model_name, dataset_name)
    print(f"Saved model to {to_local_path(save_loc)}")


def get_params_range(model):
    model_params = OPTIMIZE_PARAMS
    model_params['parameters'] = {k: v for k, v in model.params_range.items() if k not in set('default')}
    return model_params


@main.command(name="optimize", help="Optimize model")
@click.option("--comet-name", prompt="What would you like to call these experiments in comet?", default=f"model-{str(datetime.now().strftime('%m%d.%H%M'))}")
@click.option("--max-n", prompt="Maximum number of experiments: ", default=0)
@click.option("--batch-size", prompt="Batch size", default=DEFAULT_BATCH_SIZE, type=click.IntRange(min=1))
@click.option("--n-epochs", prompt="Number of epochs", default=DEFAULT_N_EPOCHS, type=click.IntRange(min=1))
def optimize(comet_name, max_n, batch_size, n_epochs):
    dataset_name, dataset_config, model, class_name = initialize_model()
    #n_epochs = prompt_num_epochs()
    #batch_size = prompt_batch_size()
    params_range = get_params_range(model)
    params_range['spec']['maxCombo'] = int(max_n)
    optimizer = Optimizer(params_range, api_key=COMET_KEY)

    for experiment in optimizer.get_experiments(project_name=PROJECT_NAME):
        experiment.set_name(comet_name)
        experiment.add_tag("optimizer_experiment")
        model_exp = model
        p = {k: experiment.get_parameter(k) for k in params_range['parameters'].keys()}
        model_exp.params = p
        model_exp = train_model(model, dataset_name, dataset_config, batch_size, n_epochs, compile_dict=COMPILE_DICT)
        loss = model_exp.test_results[0]
        experiment.log_metric("loss", loss)





if __name__ == "__main__":
    main()
