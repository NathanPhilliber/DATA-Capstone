from utils import *
from glob import glob
import os
import inspect
from networks.models import lstm_model_1
import importlib


def main():
    module_tups = get_modules(MODELS_DIR)
    selection, class_name = prompt_model_selection(module_tups)
    module, package_name = module_tups[selection]
    model_class = getattr(module, class_name)

    model = model_class(10, 1001, 5)


def prompt_model_selection(module_tups):
    list_i = 0
    names = []
    print(f"The following models were found in {to_local_path(MODELS_DIR)}:")
    for module_i, (module, module_name) in enumerate(module_tups):
        classes = get_classes(module, module_name)

        for class_i, class_name in enumerate(classes):
            print(f"  {list_i}:\t {class_name}")
            list_i += 1
            names.append(class_name)

    print()

    selection = int(input("Select model to run: "))

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
