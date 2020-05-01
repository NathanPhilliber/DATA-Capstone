import os
from utils import *
import pickle
import json
import click
from datagen.loadmatlab import mat_to_spectra


"""
This is a 'quick and dirty' script to split a sharded dataset into a subset
"""

@click.command()
@click.option('--set-name', prompt='Name of dataset to crop from')
@click.option('--new-set-name', prompt='Name of where to save new dataset')
@click.option('--shard-size', type=int, prompt='How many spectra to put in each shard')
@click.option('--action', type=str, prompt='"crop" or "reclass" or "convert"')
def main(set_name, new_set_name, shard_size, action):
    dataset_path = os.path.join(DATA_DIR, set_name)
    new_dataset_path = os.path.join(DATA_DIR, new_set_name)

    save_classes = [3, 4]
    class_groups = {1: [1, 2],
                    2: [3, 4]}

    if action == 'crop':
        print("Saving classes:", save_classes)
        crop_dataset(dataset_path=dataset_path, save_classes=save_classes, new_dataset_path=new_dataset_path, shard_size=shard_size)
    if action == 'reclass':
        print("Saving groups:", class_groups)
        reclass_dataset(dataset_path=dataset_path, class_groups=class_groups, new_dataset_path=new_dataset_path,
                        shard_size=shard_size)
    if action == 'convert':
        print("Converting matlab files")
        matlab_path = os.path.join(DATA_ROOT, "matlab", set_name)
        convert_matlab_collection(matlab_collection_path=matlab_path, new_dataset_path=new_dataset_path, shard_size=shard_size)


def convert_matlab_collection(matlab_collection_path, new_dataset_path, shard_size):
    if not os.path.exists(matlab_collection_path):
        print(f"{matlab_collection_path} does not exist.")
    files = os.listdir(matlab_collection_path)
    files = [os.path.join(matlab_collection_path, file) for file in files]

    set_name = os.path.splitext(os.path.basename(new_dataset_path))[0]

    spectras = []
    for matfile in files:
        print(f"Converting mat file: {matfile}")
        spectrum = mat_to_spectra(matfile)
        spectras.append(spectrum)

    os.mkdir(new_dataset_path)

    total_saved = 0
    shard_num = 0
    while len(spectras) >= shard_size:
            save_data = spectras[:shard_size]
            spectras = spectras[shard_size:]

            shard_num += 1
            total_saved += len(save_data)

            pickle.dump(save_data, open(
                os.path.join(new_dataset_path, f"{TRAIN_DATASET_PREFIX}_{set_name}-p{shard_num}.{DATASET_FILE_TYPE}"),
                "wb"))
            print(f"Saved shard #{shard_num} with {len(save_data)} spectra.")

    if len(spectras) > 0:
        total_saved += len(spectras)
        pickle.dump(spectras, open(
            os.path.join(new_dataset_path, f"{TRAIN_DATASET_PREFIX}_{set_name}-p{shard_num}.{DATASET_FILE_TYPE}"),
            "wb"))
        print(f"Saved final shard #{shard_num} with {len(spectras)} spectra.")



def reclass_dataset(dataset_path, class_groups, new_dataset_path, shard_size):
    """
    Transform the classes in a dataset.
    Example:
        [1,2] -> 1
        [3,4] -> 2

    :param dataset_path:
    :param class_groups: dict: {<new_class>: [<old_class>, <old_class>], ...}
    :param new_dataset_path:
    :param shard_size:
    """

    if not os.path.exists(dataset_path):
        print(f"{dataset_path} does not exist.")
    files = os.listdir(dataset_path)

    if DATAGEN_CONFIG in files:
        files.remove(DATAGEN_CONFIG)

    set_name = os.path.splitext(os.path.basename(dataset_path))[0]

    train_files = []
    test_files = []
    for myfile in files:
        if TRAIN_DATASET_PREFIX in myfile:
            train_files.append(os.path.join(dataset_path, myfile))
        elif TEST_DATASET_PREFIX in myfile:
            test_files.append(os.path.join(dataset_path, myfile))
        else:
            print(f"Unknown file: {myfile}")

    assert len(files) == len(train_files) + len(test_files), "Missing a shard file"
    train_files.sort()
    test_files.sort()

    gen_info = json.load(open(os.path.join(dataset_path, DATAGEN_CONFIG), "rb"))
    os.mkdir(new_dataset_path)

    total_instances = 0

    train_data = []
    train_saved = 0
    for train_i, train_file in enumerate(train_files):
        print(f"Processing {train_file}")
        data_all = pickle.load(open(train_file, "rb"))
        data = []

        # Filter out undesired data here
        for spectra in data_all:
            num_peaks = spectra['n']

            for new_class, old_classes in class_groups.items():

                if num_peaks in old_classes:
                    spectra["n"] = new_class
                    data.append(spectra)
                    total_instances += 1

        print(f"Filtered out {len(data_all) - len(data)} spectra.")
        train_data.extend(data)

        while len(train_data) >= shard_size:
            train_saved += 1
            save_data = train_data[:shard_size]
            train_data = train_data[shard_size:]
            pickle.dump(save_data, open(
                os.path.join(new_dataset_path, f"{TRAIN_DATASET_PREFIX}_{set_name}-p{train_saved}.{DATASET_FILE_TYPE}"),
                "wb"))
            print(f"Saved training shard #{train_saved} with {len(save_data)} spectra.")

    if len(train_data) > 0:
        train_saved += 1
        pickle.dump(train_data, open(
            os.path.join(new_dataset_path, f"{TRAIN_DATASET_PREFIX}_{set_name}-p{train_saved}.{DATASET_FILE_TYPE}"), "wb"))
        print(f"Saved final training shard #{train_saved} with {len(train_data)} spectra.")

    test_data = []
    test_saved = 0
    for test_i, test_file in enumerate(test_files):
        print(f"Processing {test_file}")
        data_all = pickle.load(open(test_file, "rb"))
        data = []

        # Filter out undesired data here
        for spectra in data_all:
            num_peaks = spectra['n']

            for new_class, old_classes in class_groups.items():
                if num_peaks in old_classes:
                    spectra["n"] = new_class
                    data.append(spectra)
                    total_instances += 1

        print(f"Filtered out {len(data_all) - len(data)} spectra.")
        test_data.extend(data)

        while len(test_data) >= shard_size:
            test_saved += 1
            save_data = test_data[:shard_size]
            test_data = test_data[shard_size:]
            pickle.dump(save_data, open(
                os.path.join(new_dataset_path, f"{TEST_DATASET_PREFIX}_{set_name}-p{test_saved}.{DATASET_FILE_TYPE}"),
                "wb"))
            print(f"Saved testing shard #{test_saved} with {len(save_data)} spectra.")

    if len(test_data) > 0:
        test_saved += 1
        pickle.dump(test_data, open(
            os.path.join(new_dataset_path, f"{TEST_DATASET_PREFIX}_{set_name}-p{test_saved}.{DATASET_FILE_TYPE}"), "wb"))
        print(f"Saved final testing shard #{test_saved} with {len(test_data)} spectra.")

    print("Writing config")
    gen_info["num_instances"] = total_instances
    json.dump(gen_info, open(os.path.join(new_dataset_path, DATAGEN_CONFIG), "w"))

    print("Done")


def crop_dataset(dataset_path, save_classes, new_dataset_path, shard_size):
    if not os.path.exists(dataset_path):
        print(f"{dataset_path} does not exist.")
    files = os.listdir(dataset_path)

    if DATAGEN_CONFIG in files:
        files.remove(DATAGEN_CONFIG)

    set_name = os.path.splitext(os.path.basename(dataset_path))[0]

    train_files = []
    test_files = []
    for myfile in files:
        if TRAIN_DATASET_PREFIX in myfile:
            train_files.append(os.path.join(dataset_path, myfile))
        elif TEST_DATASET_PREFIX in myfile:
            test_files.append(os.path.join(dataset_path, myfile))
        else:
            print(f"Unknown file: {myfile}")

    assert len(files) == len(train_files) + len(test_files), "Missing a shard file"
    train_files.sort()
    test_files.sort()

    gen_info = json.load(open(os.path.join(dataset_path, DATAGEN_CONFIG), "rb"))
    os.mkdir(new_dataset_path)

    total_instances = 0

    train_data = []
    train_saved = 0
    for train_i, train_file in enumerate(train_files):
        print(f"Processing {train_file}")
        data_all = pickle.load(open(train_file, "rb"))
        data = []

        # Filter out undesired data here
        for spectra in data_all:
            num_peaks = spectra['n']

            if num_peaks in save_classes:
                data.append(spectra)
                total_instances += 1

        print(f"Filtered out {len(data_all) - len(data)} spectra.")
        train_data.extend(data)

        while len(train_data) >= shard_size:
            train_saved += 1
            save_data = train_data[:shard_size]
            train_data = train_data[shard_size:]
            pickle.dump(save_data, open(
                os.path.join(new_dataset_path, f"{TRAIN_DATASET_PREFIX}_{set_name}-p{train_saved}.{DATASET_FILE_TYPE}"),
                "wb"))
            print(f"Saved training shard #{train_saved} with {len(save_data)} spectra.")

    if len(train_data) > 0:
        train_saved += 1
        pickle.dump(train_data, open(
            os.path.join(new_dataset_path, f"{TRAIN_DATASET_PREFIX}_{set_name}-p{train_saved}.{DATASET_FILE_TYPE}"), "wb"))
        print(f"Saved final training shard #{train_saved} with {len(train_data)} spectra.")

    test_data = []
    test_saved = 0
    for test_i, test_file in enumerate(test_files):
        print(f"Processing {test_file}")
        data_all = pickle.load(open(test_file, "rb"))
        data = []

        # Filter out undesired data here
        for spectra in data_all:
            num_peaks = spectra['n']

            if num_peaks in save_classes:
                data.append(spectra)
                total_instances += 1

        print(f"Filtered out {len(data_all) - len(data)} spectra.")
        test_data.extend(data)

        while len(test_data) >= shard_size:
            test_saved += 1
            save_data = test_data[:shard_size]
            test_data = test_data[shard_size:]
            pickle.dump(save_data, open(
                os.path.join(new_dataset_path, f"{TEST_DATASET_PREFIX}_{set_name}-p{test_saved}.{DATASET_FILE_TYPE}"),
                "wb"))
            print(f"Saved testing shard #{test_saved} with {len(save_data)} spectra.")

    if len(test_data) > 0:
        test_saved += 1
        pickle.dump(test_data, open(
            os.path.join(new_dataset_path, f"{TEST_DATASET_PREFIX}_{set_name}-p{test_saved}.{DATASET_FILE_TYPE}"), "wb"))
        print(f"Saved final testing shard #{test_saved} with {len(test_data)} spectra.")

    print("Writing config")
    gen_info["num_instances"] = total_instances
    json.dump(gen_info, open(os.path.join(new_dataset_path, DATAGEN_CONFIG), "w"))

    print("Done")


if __name__ == "__main__":
    main()
