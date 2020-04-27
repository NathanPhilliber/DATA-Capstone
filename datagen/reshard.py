from utils import *
import click
import shutil
import pickle

temp_name = "temp-savespace"

@click.command()
@click.option('--set-name', prompt='Name of dataset to modify')
@click.option('--shard-size', type=int, prompt='New size of shard')
def main(set_name, shard_size):
    data_dir = os.path.join(DATA_DIR, set_name)
    temp_data_dir = os.path.join(DATA_DIR, set_name, temp_name)
    if not os.path.exists(data_dir):
        print(f"{data_dir} does not exist.")
    files = os.listdir(data_dir)

    if DATAGEN_CONFIG in files:
        files.remove(DATAGEN_CONFIG)
    if temp_name in files:
        print(f"Deleting {temp_data_dir}")
        shutil.rmtree(temp_data_dir)
        files.remove(temp_name)

    train_files = []
    test_files = []
    for myfile in files:
        if TRAIN_DATASET_PREFIX in myfile:
            train_files.append(os.path.join(data_dir, myfile))
        elif TEST_DATASET_PREFIX in myfile:
            test_files.append(os.path.join(data_dir, myfile))
        else:
            print(f"Unknown file: {myfile}")

    assert len(files) == len(train_files) + len(test_files), "Missing a shard file"
    train_files.sort()
    test_files.sort()

    os.mkdir(temp_data_dir)

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
            pickle.dump(save_data, open(os.path.join(temp_data_dir, f"{TRAIN_DATASET_PREFIX}_{set_name}-p{train_saved}.{DATASET_FILE_TYPE}"), "wb"))
            print(f"Saved training shard #{train_saved} with {len(save_data)} spectra.")

    if len(train_data) > 0:
        train_saved += 1
        pickle.dump(train_data, open(os.path.join(temp_data_dir, f"{TRAIN_DATASET_PREFIX}_{set_name}-p{train_saved}.{DATASET_FILE_TYPE}"), "wb"))
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
            pickle.dump(save_data, open(os.path.join(temp_data_dir, f"{TEST_DATASET_PREFIX}_{set_name}-p{test_saved}.{DATASET_FILE_TYPE}"), "wb"))
            print(f"Saved testing shard #{test_saved} with {len(save_data)} spectra.")

    if len(test_data) > 0:
        test_saved += 1
        pickle.dump(test_data, open(os.path.join(temp_data_dir, f"{TEST_DATASET_PREFIX}_{set_name}-p{test_saved}.{DATASET_FILE_TYPE}"), "wb"))
        print(f"Saved final testing shard #{test_saved} with {len(test_data)} spectra.")

    print("Removing old shards")
    for myfile in files:
        myfile = os.path.join(data_dir, myfile)
        os.remove(myfile)
        print(f"  Removed {myfile}.")

    new_shards = os.listdir(temp_data_dir)
    for new_shard in new_shards:
        shutil.move(os.path.join(temp_data_dir, new_shard), data_dir)

    os.removedirs(temp_data_dir)

    print("Done")


if __name__ == "__main__":
    main()
