from utils import *
import os
import inspect
import importlib
import json
from networks.SpectraPreprocessor import SpectraPreprocessor
from datetime import datetime


GENERATOR_LIMIT = 10000  # The minimum number of data points where fit generator should be used


def main():
    module_tups = get_modules(MODELS_DIR)
    model_selection, class_name = prompt_model_selection(module_tups)
    module, package_name = module_tups[model_selection]
    model_class = getattr(module, class_name)

    dataset_name = prompt_dataset_selection()
    dataset_config = json.load(open(os.path.join(DATA_DIR, dataset_name, DATAGEN_CONFIG), "r"))

    n_epochs = prompt_num_epochs()
    batch_size = prompt_batch_size()

    model = model_class(dataset_config["num_channels"], 1001, 5)
    use_generator = dataset_config["num_instances"] >= GENERATOR_LIMIT
    spectra_pp = SpectraPreprocessor(dataset_name=dataset_name, use_generator=use_generator)

    baseline_model_compile_dict = {'optimizer': 'adam',
                                   'loss': 'categorical_crossentropy',
                                   'metrics': ['accuracy', 'mae', 'mse']}

    if use_generator:
        print("Using fit generator.")
        X_test, y_test = spectra_pp.transform_test(encoded=True)
        model.fit_generator(spectra_pp, spectra_pp.datagen_config["num_instances"], X_test,
                            y_test, batch_size=batch_size, epochs=n_epochs,
                            compile_dict=baseline_model_compile_dict, encoded=True)

    else:
        X_train, y_train, X_test, y_test = spectra_pp.transform(encoded=True)
        model.fit(X_train, y_train, X_test, y_test, batch_size=batch_size, epochs=n_epochs,
                  compile_dict=baseline_model_compile_dict)

    model.keras_model.save("%s.h5" % os.path.join(MODEL_RES_DIR, model_class + "-" + str(datetime.now())))


def prompt_batch_size():
    return int(input("Enter batch size: "))


def prompt_num_epochs():
    return int(input("Enter number of epochs to train for: "))


def prompt_dataset_selection():
    data_dirs = os.listdir(DATA_DIR)
    print(f"\nThe following datasets were found in {to_local_path(DATA_DIR)}:")
    for dir_i, dir in enumerate(data_dirs):
        print(f"  {dir_i}:\t {dir}")

    selection = int(input("\nSelect dataset to use: "))

    return data_dirs[selection]


def prompt_model_selection(module_tups):
    list_i = 0
    names = []

    print(f"\nThe following models were found in {to_local_path(MODELS_DIR)}:")
    for module_i, (module, module_name) in enumerate(module_tups):
        classes = get_classes(module, module_name)

        for class_i, class_name in enumerate(classes):
            print(f"  {list_i}:\t {class_name}")
            list_i += 1
            names.append(class_name)

    selection = int(input("\nSelect model to run: "))

    return selection, names[selection]


def get_classes(module, package_name):
    return [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == package_name]


def get_modules(dirpath):
    mods = []
    for file in os.listdir(dirpath):
        if file[:2] != "__":
            localpath = to_local_path(dirpath)
            package_name = ".".join(os.path.split(localpath)) + "." + os.path.splitext(file)[0]
            mod = importlib.import_module(package_name)

            mods.append((mod, package_name))

    return mods


if __name__ == "__main__":
    main()
