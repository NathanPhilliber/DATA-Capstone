from utils import *
from datagen.spectra_generator import SpectraGenerator
from datagen.spectra_loader import SpectraLoader
import click
import math
import os
import math


MAX_REC_SHARD_SIZE = 10000
NUM_EXAMPLE_IMAGES = 10


def prompt_matlab_script():
    scripts = sorted(os.listdir(GEN_DIR))
    scripts_prompt = f"Select from the following MATLAB scripts, located in: {to_local_path(GEN_DIR)}"

    for script_i, script_name in enumerate(scripts):
        scripts_prompt += f"\n{script_i}: {script_name}"

    return scripts_prompt


def get_matlab_selection(num_script):
    scripts = sorted(os.listdir(GEN_DIR))

    return scripts[num_script]


def save_images(dataset_dir, spectra_loader, num_examples):
    img_dir = 'imgs'
    img_directory = os.path.join(dataset_dir, img_dir)
    if img_dir not in os.listdir(dataset_dir):
        try_create_directory(img_directory)

    spectra_loader.save_spectra_imgs(img_directory, num_examples)


@click.command()
@click.option('--name', prompt=f'Spectra are stored in this directory --  {to_local_path(DATA_DIR)}/')
@click.option('--version', type=int, default=0, prompt=prompt_matlab_script() + "   ")
@click.option('--num-instances', type=int, default=10000, prompt='Number of instances to create')
@click.option('--shard-size', type=int, default=0, prompt='How many spectra to put in each shard (0 = no shard)')
@click.option('--num-channels', type=float, prompt=f'Number of channels to generate', default=SpectraGenerator.DEFAULT_NC)
@click.option('--n-max', type=float, prompt=f'Maximum number of modes', default=SpectraGenerator.DEFAULT_N_MAX)
@click.option('--n-max-s', type=float, prompt=f'Maximum number of shell peaks', default=SpectraGenerator.DEFAULT_N_MAX_S)
@click.option('--scale', type=float, prompt=f'Scale or width of window', default=SpectraGenerator.DEFAULT_SCALE)
@click.option('--omega-shift', type=float, prompt=f'Omega Shift', default=SpectraGenerator.DEFAULT_OMEGA_SHIFT)
@click.option('--dg', type=float, prompt=f'Variation of Gamma', default=SpectraGenerator.DEFAULT_DG)
@click.option('--dgs', type=float, prompt=f'Gamma variation of shell modes', default=SpectraGenerator.DEFAULT_DGS)
def main(name, version, num_instances, shard_size, num_channels, n_max, n_max_s, scale, omega_shift, dg, dgs):

    # Setup data directory
    matlab_script = get_matlab_selection(version)
    directory = os.path.join(DATA_DIR, name)
    try_create_directory(directory)
    check_clear_directory(directory)

    print("Creating generator...")
    spectra_generator = SpectraGenerator(matlab_script=matlab_script, nc=num_channels, n_max=n_max, n_max_s=n_max_s, scale=scale,
                                         omega_shift=omega_shift, dg=dg, dgs=dgs)

    # If we don't want to shard, set to num_instances to make num_shards = 1
    if shard_size == 0:
        shard_size = num_instances
    num_shards = int(math.ceil(num_instances/shard_size))

    if shard_size >= MAX_REC_SHARD_SIZE:
        print("Warning! This dataset is large, consider using smaller shards ('--shard-size')")
    if num_shards > 1:
        print(f"Saving training data into {num_shards} shards.")

    num_train_shards_saved = 0
    num_test_shards_saved = 0
    num_saved = 0
    num_gen = 0
    train_set_buffer = []
    test_set_buffer = []
    for shard_i in range(0, num_shards):
        num_left = num_instances - num_gen
        gen_num = shard_size

        if num_left < shard_size:
            gen_num = num_left

        print(f"\nGenerating {gen_num} spectra for shard #{shard_i+1} ({num_left} left)...")
        spectra_json = spectra_generator.generate_spectra_json(gen_num)

        print("  Making SpectraLoader...")
        spectra_loader = SpectraLoader(spectra_json=spectra_json)
        save_images(directory, spectra_loader, math.ceil(NUM_EXAMPLE_IMAGES/num_shards))

        print(f"  Splitting data...")
        spectra_train, spectra_test = spectra_loader.spectra_train_test_splitter()
        train_set_buffer.extend([spectrum.__dict__ for spectrum in spectra_train])
        test_set_buffer.extend([spectrum.__dict__ for spectrum in spectra_test])
        num_gen += len(spectra_train) + len(spectra_test)
        print(f"    {len(spectra_train)} Train, {len(spectra_test)} Test")

        while len(train_set_buffer) >= shard_size or (shard_i == num_shards - 1 and len(train_set_buffer) > 0):
            print("  Saving training data...")
            spectra_train_json = train_set_buffer[:shard_size]
            train_set_buffer = train_set_buffer[shard_size:]
            train_name = f'{TRAIN_DATASET_PREFIX}_{name}.pkl' if shard_size == num_instances else \
                f'{TRAIN_DATASET_PREFIX}_{name}-p{num_train_shards_saved + 1}.{DATASET_FILE_TYPE}'

            SpectraGenerator.save_spectra(spectra_train_json, train_name, directory)
            print(f"    Saved {len(spectra_train_json)} spectra")
            num_train_shards_saved += 1
            num_saved += len(spectra_train_json)

        while len(test_set_buffer) >= shard_size or (shard_i == num_shards - 1 and len(test_set_buffer) > 0):
            print("  Saving testing data...")
            spectra_test_json = test_set_buffer[:shard_size]
            test_set_buffer = test_set_buffer[shard_size:]
            test_name = f'{TEST_DATASET_PREFIX}_{name}.pkl' if shard_size == num_instances else \
                f'{TEST_DATASET_PREFIX}_{name}-p{num_test_shards_saved + 1}.{DATASET_FILE_TYPE}'

            SpectraGenerator.save_spectra(spectra_test_json, test_name, directory)
            print(f"    Saved {len(spectra_test_json)} spectra")
            num_test_shards_saved += 1
            num_saved += len(spectra_test_json)

    print("\nSaving info...")
    spectra_generator.create_spectra_info(num_instances, directory)
    print(f"Saved {num_saved} spectra to {directory}.\nDone.")


if __name__ == '__main__':
    main()
