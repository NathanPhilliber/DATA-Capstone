from utils import *
import os
import inspect
import importlib
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
#tf.logging.set_verbosity(tf.logging.ERROR)

click_utils = click.Group()

@click.group()
def main():
    pass


def get_module(model_name=None):
    module_tups = get_modules(NETWORKS_DIR)
    model_selection, class_name = prompt_model_selection(module_tups, model_name)
    module, package_name = module_tups[model_selection]

    return module, class_name, package_name


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


def set_result_dir(class_name):
    result_dir = prompt_result_selection(class_name)
    result_info = json.load(open(os.path.join(result_dir, TRAIN_INFO_FILENAME), "rb"))
    return result_dir, result_info


def initialize_model(dataset_name, model_name=None):
    dataset_config = load_dataset_info(dataset_name)
    module, class_name, package_name = get_module(model_name)
    model = load_model(module, class_name, dataset_config['num_channels'], dataset_config['n_max'], dataset_config['num_timesteps'])
    return dataset_config, model, class_name


def train_model(model, dataset_name, dataset_config, batch_size, n_epochs, compile_dict=None):
    use_generator = dataset_config["num_instances"] > GENERATOR_LIMIT
    spectra_pp = SpectraPreprocessor(dataset_name=dataset_name, use_generator=use_generator)
    model.log_imgs(dataset_name)
    model.log_script(dataset_config)

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

    msg += f"\nThe following datasets were found in {to_local_path(DATA_DIR)}:"
    msg += f"{'Selection':10} {'Set Name':15} {'Num Spectra':15} {'Num Channels':15}"
    for dir_i, dir_name in enumerate(data_dirs):
        config = SpectraLoader.read_dataset_config(dir_name)
        msg += f"  {dir_i:6}:  {dir_name:15} {format(config['num_instances'], ','):15} {int(config['num_channels']):2}"

    msg += "\nSelect dataset to use: "

    return msg


def get_dataset_name(ctx, param, dataset_name_or_selection):
    data_dirs = sorted(os.listdir(DATA_DIR))
    data_dirs = [data_dir for data_dir in data_dirs if os.path.isdir(os.path.join(DATA_DIR, data_dir))]

    try:
        selection = int(dataset_name_or_selection)
        if selection > len(data_dirs) or selection < 0:
            raise Exception("Invalid option: %d out of range" % selection)

        dataset_name = data_dirs[selection]
    except Exception:
        dataset_name = dataset_name_or_selection

    if dataset_name not in data_dirs:
        raise Exception("Could not find dataset with set_name='%s' in '%s" % (dataset_name, DATA_DIR))

    ctx.params["dataset_name"] = dataset_name
    return dataset_name


@main.command(name="evaluate")
def evaluate_model():
    click.clear()
    print("Train Existing Model Setup\n")

    dataset_name, dataset_config, model, class_name = initialize_model()
    result_dir, result_info = set_result_dir(class_name)

    model.persist(os.path.basename(result_dir))
    y_true, y_pred = model.preds
    labels = [str(i) for i in range(1, int(dataset_config['n_max'] + 1))]

    print(classification_report(y_true, y_pred, target_names=labels))


@main.command(name="continue", help="Continue training an existing run")
@click.option("--n-epochs", prompt="Number of epochs", default=DEFAULT_N_EPOCHS, type=click.IntRange(min=1))
def continue_train_model(n_epochs):
    click.clear()
    print("Train Existing Model Setup\n")

    dataset_name, dataset_config, model, class_name = initialize_model()
    result_dir, result_info = set_result_dir(class_name)

    model.persist(os.path.basename(result_dir))
    #n_epochs = prompt_num_epochs()
    model = train_model(model, dataset_name, dataset_config, model.batch_size, n_epochs)

    y_true, y_pred = model.preds
    labels = [str(i) for i in range(1, int(dataset_config['n_max'] + 1))]
    model.experiment.log_confusion_matrix(y_true, y_pred, labels=labels)
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    print(classification_report(y_true, y_pred, target_names=[1, 2, 3, 4]))

    save_loc = model.save(class_name, dataset_name)
    print(f"Saved model to {to_local_path(save_loc)}")


@main.command(name="new", help="Train a new model")
@click.option("--comet-name", "-cn", prompt="What would you like to call this run on comet?", default=f"model-{str(datetime.now().strftime('%m%d.%H%M'))}")
@click.option("--batch-size", "-bs", prompt="Batch size", default=DEFAULT_BATCH_SIZE, type=click.IntRange(min=1))
@click.option("--n-epochs", "-n", prompt="Number of epochs", default=DEFAULT_N_EPOCHS, type=click.IntRange(min=1))
@click.option('--dataset-name', "-d", prompt=prompt_dataset_string(), default=None, callback=get_dataset_name)
@click.option('--model-name', "-m", default=None)
@click.option('--use-comet', "-uc", default=True)
def train_new_model(comet_name, batch_size, n_epochs, dataset_name, model_name, use_comet):
    #dataset_name = prompt_dataset_selection(dataset_name)
    print("dataset name:", dataset_name)
    dataset_config, model, class_name = initialize_model(dataset_name, model_name)

    if use_comet:
        model.load_comet_new(comet_name, dataset_config)

    model = train_model(model, dataset_name, dataset_config, batch_size, n_epochs, compile_dict=COMPILE_DICT)
    model.experiment.log_parameters(model.get_info_dict())

    y_true, y_pred = model.preds
    #model.experiment.log_confusion_matrix(y_true, y_pred)
    labels = [str(i) for i in range(1, int(dataset_config['n_max'] + 1))]
    model.experiment.log_confusion_matrix(y_true, y_pred, labels=labels)

    save_loc = model.save(class_name, dataset_name)
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


#def prompt_batch_size():
#    return int(input("Enter batch size: "))


#def prompt_num_epochs():
#    return int(input("Enter number of epochs to train for: "))









def prompt_model_selection(module_tups, model_name=None):
    list_i = 0
    names = []
    module_indices = []

    for module_i, (module, module_name) in enumerate(module_tups):
        classes = sorted(get_classes(module, module_name))

        for class_i, class_name in enumerate(classes):
            list_i += 1
            names.append(class_name)
            module_indices.append(module_i)

    if model_name is None:
        print(f"\nThe following models were found in {to_local_path(NETWORKS_DIR)}:")
        for sel_i, class_name in enumerate(names):
            print(f"  {sel_i}:\t {class_name}")

        selection = int(input("\nSelect model to run: "))
    else:
        if model_name in names:
            selection = names.index(model_name)
        else:
            raise Exception("Could not find model with model_name='%s' in '%s'" % (model_name, NETWORKS_DIR))

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
