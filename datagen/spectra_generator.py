from utils import *
from datagen.SpectraGenerator import SpectraGenerator
from datagen.SpectraLoader import SpectraLoader
import click
import math


@click.command()
@click.option('--num-instances', default=10000, help='Number of instances to create. ')
@click.option('--name', prompt='Spectra are stored in this directory. ')
@click.option('--shard-size', default=0, help='How many spectra to put in each shard (0 = no shard). ')
def main(num_instances, name, shard_size):

    # Setup data directory
    directory = os.path.join(DATA_DIR, name)
    try_create_directory(directory)
    check_clear_directory(directory)

    print("Creating generator")
    spectra_generator = SpectraGenerator()

    # If we don't want to shard, set to num_instances to make num_shards = 1
    if shard_size == 0:
        shard_size = num_instances
    num_shards = int(math.ceil(num_instances/shard_size))

    num_saved = 0
    for shard_i in range(0, num_shards):
        num_left = num_instances - num_saved
        gen_num = shard_size

        if num_left < shard_size:
            gen_num = num_left

        print(f"Generating {gen_num} spectra for shard #{shard_i+1}...")
        spectra_json = spectra_generator.generate_spectra_json(gen_num)

        print("  Making SpectraLoader...")
        spectra_loader = SpectraLoader(spectra_json=spectra_json)

        print("  Splitting data...")
        spectra_train, spectra_test = spectra_loader.spectra_train_test_splitter()
        spectra_train_json = [spectrum.__dict__ for spectrum in spectra_train]
        spectra_test_json = [spectrum.__dict__ for spectrum in spectra_test]

        print("  Saving training data...")
        train_name = f'{TRAIN_DATASET_PREFIX}_{name}.pkl' if shard_size == num_instances else \
            f'{TRAIN_DATASET_PREFIX}_{name}-p{shard_i+1}.{DATASET_FILE_TYPE}'
        SpectraGenerator.save_spectra(spectra_train_json, train_name, directory)

        print("  Saving testing data...")
        test_name = f'{TEST_DATASET_PREFIX}_{name}.pkl' if shard_size == num_instances else \
            f'{TEST_DATASET_PREFIX}_{name}-p{shard_i + 1}.{DATASET_FILE_TYPE}'
        SpectraGenerator.save_spectra(spectra_test_json, test_name, directory)

        num_saved += gen_num

    print("Saving info...")
    spectra_generator.create_spectra_info(num_instances, directory)
    print(f"Saved {num_saved} spectra.\nDone.")


if __name__ == '__main__':
    main()
