import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

GEN_DIR = os.path.join(PROJECT_ROOT, "datagen")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NETWORKS_DIR = os.path.join(MODELS_DIR, "networks")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_DIR = os.path.join(DATA_ROOT, "datasets")
MODEL_RES_DIR = os.path.join(DATA_ROOT, "results")

TRAIN_DATASET_PREFIX = "train"
TEST_DATASET_PREFIX = "test"
DATASET_FILE_TYPE = "pkl"
DATAGEN_CONFIG = "gen_info.json"
WEIGHTS_FILENAME = "weights.h5"
TRAIN_INFO_FILENAME = "info.json"
COMET_KEY = "rKj0YN2SYHxxZ5dYvS3WJ1jkz"

PROJECT_NAME = 'data-capstone-nasa'


def to_local_path(path):
    return path.replace(PROJECT_ROOT, "")[len(os.path.sep):]


def try_create_directory(directory, silent=False):
    """
    Create directory if it doesn't already exist

    :param directory: dir path
    :param silent: bool, if True then do not print messages
    :returns bool, True if dir was created
    """

    try:
        os.mkdir(directory)
        return True
    except FileExistsError:
        if not silent:
            print(f"Info: '{directory}' already exists.")
        return False


def check_clear_directory(directory, force=False):
    """
    Check if there are files in a directory. If there are prompt to delete
    :param directory: dir to check
    :param force: do not prompt for deletion
    :return: True if directory is clear, False otherwise
    """
    files = os.listdir(directory)

    if len(files) == 0:
        return True

    answer = 'y'
    if not force:
        print(f"Warning '{directory}' contains the following files:")
        print("\t" + "\n\t".join(files))

        answer = input("\nWould you like to remove them? (y/n) ")

    if answer.lower() == 'y':
        for file in files:
            filepath = os.path.join(directory, file)
            os.remove(filepath)
        print(f"Deleted {len(files)} files.")
        return True

    return False


try_create_directory(DATA_ROOT, silent=True)
try_create_directory(DATA_DIR, silent=True)
try_create_directory(MODEL_RES_DIR, silent=True)

