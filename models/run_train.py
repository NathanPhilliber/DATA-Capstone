from utils import *
from comet_ml import Experiment, Optimizer
import os
import inspect
import importlib
import json
from models.spectra_preprocessor import SpectraPreprocessor
from datagen.spectra_loader import SpectraLoader
from datetime import datetime
import click
import tensorflow as tf

COMPILE_DICT = {'optimizer': 'adam','loss': 'categorical_crossentropy', 'metrics': ['accuracy', 'mae', 'mse']}
PROJECT_NAME = 'data-capstone-nasa'
OPTIMIZE_PARAMS = {'algorithm': 'bayes', 'spec': {'metric': 'loss', 'objective': 'minimize'}}

GENERATOR_LIMIT = 10000  # The minimum number of data points where fit generator should be used
tf.logging.set_verbosity(tf.logging.ERROR)


@click.group()
def main():
    pass


def load_model(num_channels, n_max, num_timesteps):
    module_tups = get_modules(NETWORKS_DIR)
    model_selection, class_name = prompt_model_selection(module_tups)
    module, package_name = module_tups[model_selection]
    model_class = getattr(module, class_name)
    model = model_class(num_channels, num_timesteps, n_max)
    return model, class_name


def load_dataset_info():
    dataset_name = prompt_dataset_selection()
    dataset_config = json.load(open(os.path.join(DATA_DIR, dataset_name, DATAGEN_CONFIG), "r"))
    return dataset_name, dataset_config


def load_data(dataset_name, dataset_config):
    use_generator = dataset_config["num_instances"] > GENERATOR_LIMIT
    spectra_pp = SpectraPreprocessor(dataset_name=dataset_name, use_generator=use_generator)
    return spectra_pp


def set_result_dir(class_name):
    result_dir = prompt_result_selection(class_name)
    result_info = json.load(open(os.path.join(result_dir, TRAIN_INFO_FILENAME), "rb"))
    return result_dir, result_info


def initialize_model():
    dataset_name, dataset_config = load_dataset_info()
    model, class_name = load_model(dataset_config['num_channels'], dataset_config['n_max'], dataset_config['num_timesteps'])
    return dataset_name, dataset_config, model, class_name


def initialize_comet(comet_name, dataset_config):
    exp = create_comet_exp(comet_name)
    log_data_attributes(exp, dataset_config)
    exp.log_asset('datagen/spectra_generator.m')
    return exp


def train_model(model, dataset_name, dataset_config, batch_size, n_epochs, compile_dict=None):
    use_generator = dataset_config["num_instances"] > GENERATOR_LIMIT
    spectra_pp = SpectraPreprocessor(dataset_name=dataset_name, use_generator=use_generator)

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


def log_data_attributes(experiment, dataset_config):
    for key, value in dataset_config.items():
        experiment.log_parameter("SPECTRUM_" + key, value)


@main.command(name="continue", help="Continue training an existing run")
@click.option("--comet-name", prompt="What would you like to call this run on comet?", default=f"model-{str(datetime.now().strftime('%m%d.%H%M'))}")
def continue_train_model(comet_name):
    click.clear()
    print("Train Existing Model Setup\n")

    dataset_name, dataset_config, model, class_name = initialize_model()
    exp = initialize_comet(comet_name, dataset_config)
    result_dir, result_info = set_result_dir(class_name)

    model.persist(os.path.basename(result_dir))
    n_epochs = prompt_num_epochs()
    model = train_model(model, dataset_name, dataset_config, model.batch_size, n_epochs)

    save_loc = model.save(class_name, dataset_name)
    print(f"Saved model to {to_local_path(save_loc)}")


@main.command(name="new", help="Train a new model")
@click.option("--comet-name", prompt="What would you like to call this run on comet?", default=f"model-{str(datetime.now().strftime('%m%d.%H%M'))}")
def train_new_model(comet_name):
    click.clear()
    print("Train New Model Setup\n")

    dataset_name, dataset_config, model, class_name = initialize_model()
    exp = initialize_comet(comet_name, dataset_config)

    n_epochs = prompt_num_epochs()
    batch_size = prompt_batch_size()
    model = train_model(model, dataset_name, dataset_config, batch_size, n_epochs, compile_dict=COMPILE_DICT)
    exp.log_parameters(model.get_info_dict())

    y_true, y_pred = model.preds
    exp.log_confusion_matrix(y_true, y_pred)

    save_loc = model.save(class_name, dataset_name)
    print(f"Saved model to {to_local_path(save_loc)}")


def get_params_range(model):
    model_params = OPTIMIZE_PARAMS
    model_params['parameters'] = {k: v for k, v in model.params_range.items() if k not in set('default')}
    return model_params


@main.command(name="optimize", help="Optimize model")
@click.option("--comet-name", prompt="What would you like to call these experiments in comet?", default=f"model-{str(datetime.now().strftime('%m%d.%H%M'))}")
@click.option("--max-n", prompt="Maximum number of experiments: ", default=0)
def optimize(comet_name, max_n):
    dataset_name, dataset_config, model, class_name = initialize_model()
    n_epochs = prompt_num_epochs()
    batch_size = prompt_batch_size()
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


def create_comet_exp(name):
    exp = Experiment(
        api_key=COMET_KEY,
        project_name=PROJECT_NAME)
    exp.set_name(name)
    return exp


def prompt_batch_size():
    return int(input("Enter batch size: "))


def prompt_num_epochs():
    return int(input("Enter number of epochs to train for: "))


def prompt_dataset_selection():
    data_dirs = sorted(os.listdir(DATA_DIR))
    data_dirs = [data_dir for data_dir in data_dirs if os.path.isdir(os.path.join(DATA_DIR, data_dir))]
    print(f"\nThe following datasets were found in {to_local_path(DATA_DIR)}:")
    print(f"{'Selection':10} {'Set Name':15} {'Num Spectra':15} {'Num Channels':15}")
    for dir_i, dir_name in enumerate(data_dirs):
        config = SpectraLoader.read_dataset_config(dir_name)
        print(f"  {dir_i:6}:  {dir_name:15} {format(config['num_instances'], ','):15} {int(config['num_channels']):2}")

    selection = int(input("\nSelect dataset to use: "))

    return data_dirs[selection]


def prompt_model_selection(module_tups):
    list_i = 0
    names = []
    module_indices = []

    print(f"\nThe following models were found in {to_local_path(NETWORKS_DIR)}:")
    for module_i, (module, module_name) in enumerate(module_tups):
        classes = sorted(get_classes(module, module_name))

        for class_i, class_name in enumerate(classes):
            print(f"  {list_i}:\t {class_name}")
            list_i += 1
            names.append(class_name)
            module_indices.append(module_i)

    selection = int(input("\nSelect model to run: "))

    return module_indices[selection], names[selection]


def prompt_result_selection(class_name):
    result_dirs = os.listdir(MODEL_RES_DIR)
    result_dirs = sorted([result_dir for result_dir in result_dirs if class_name == os.path.basename(result_dir).split("_")[0]])

    prefix = "  "
    print("Found Existing Models:")
    for dir_i, result_dir in enumerate(result_dirs):
        print(f"  {dir_i:3}:  {result_dir}")

    selection = int(input("\nSelect model to train: "))
    return os.path.join(MODEL_RES_DIR, result_dirs[selection])


def get_classes(module, package_name):
    return [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == package_name]


def get_modules(dirpath):
    mods = []
    for file in sorted(os.listdir(dirpath)):
        if file[:2] != "__":
            localpath = to_local_path(dirpath)
            package_name = ".".join(os.path.split(localpath)) + "." + os.path.splitext(file)[0]
            mod = importlib.import_module(package_name)

            mods.append((mod, package_name))

    return mods


if __name__ == "__main__":
    main()
