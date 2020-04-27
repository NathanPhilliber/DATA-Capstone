import os
from utils import *
import pickle
import json
import click

"""
This is a 'quick and dirty' script to split a sharded dataset into a subset
"""

@click.command()
@click.option('--set-name', prompt='Name of dataset to crop from')
@click.option('--new-set-name', prompt='Name of where to save new dataset')
@click.option('--shard-size', type=int, prompt='How many spectra to put in each shard')
def main(set_name, new_set_name, shard_size):
    dataset_path = os.path.join(DATA_DIR, set_name)
    new_dataset_path = os.path.join(DATA_DIR, new_set_name)

    save_classes = [3, 4]
    print("Saving classes:", save_classes)

    crop_dataset(dataset_path=dataset_path, save_classes=save_classes, new_dataset_path=new_dataset_path, shard_size=shard_size)


def crop_dataset(dataset_path, save_classes, new_dataset_path, shard_size):
    if not os.path.exists(dataset_path):
        print(f"{dataset_path} does not exist.")
    files = os.listdir(dataset_path)

    if DATAGEN_CONFIG in files:
        files.remove(DATAGEN_CONFIG)

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
    set_name = os.path.basename(train_files[0].replace(TRAIN_DATASET_PREFIX, ""))

    os.mkdir(new_dataset_path)

    train_data = []
    train_saved = 0
    for train_i, train_file in enumerate(train_files):
        print(f"Processing {train_file}")
        data = pickle.load(open(train_file, "rb"))
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
        data = pickle.load(open(test_file, "rb"))
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

    print("Done")


if __name__ == "__main__":
    main()
