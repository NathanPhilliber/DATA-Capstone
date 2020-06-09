from utils import *
from datagen.spectra_generator import LocalSpectraGenerator, S3SpectraGenerator, SpectraGenerator
from datagen.spectra_loader import SpectraLoader
import click
import os
import math


MAX_REC_SHARD_SIZE = 10000
NUM_EXAMPLE_IMAGES = 10


def prompt_matlab_script():
    """

    :return: list[str] A list of the matlab scripts found under the `matlab_scripts` directory.
    """
    scripts = sorted(os.listdir(GEN_DIR))
    scripts_prompt = f"Select from the following MATLAB scripts, located in: {to_local_path(GEN_DIR)}"

    for script_i, script_name in enumerate(scripts):
        scripts_prompt += f"\n{script_i}: {script_name}"

    return scripts_prompt


def get_matlab_selection(num_script):
    """

    :param num_script: int Gets the matlab script of a specified integer.
    :return: str The matlab script indexed by `num_script`.
    """
    scripts = sorted(os.listdir(GEN_DIR))
    return scripts[num_script]


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
@click.option('--gamma-amp-factor', type=float, prompt=f'Gamma amp factor', default=SpectraGenerator.DEFAULT_GAMMA_AMP_FACTOR)
@click.option('--amp-factor', type=float, prompt=f'Amp factor', default=SpectraGenerator.DEFAULT_AMP_FACTOR)
@click.option('--epsilon2', type=float, prompt=f'Epsilon2', default=SpectraGenerator.DEFAULT_EPSILON2)
def main(name, version, num_instances, shard_size, num_channels, n_max, n_max_s, scale, omega_shift, dg, dgs, gamma_amp_factor,
         amp_factor, epsilon2):
    """
    Use this function in order to create spectra-data with user input through command line arguments.

    :param name: str The name for the directory where we will store the spectra data produced.
    :param version: str The selected version of the matlab script that will be used to generate data.
    :param num_instances: in The number of instances that we will generate. Note: This must be larger than 100.
    :param shard_size: int The number of shards used to store the data.
    :param num_channels: int The number of channels used to generate the data.
    :param n_max: int The maximum number of possible liquid modes per window in script used to generate data.
    :param n_max_s: int Maximum number of shell modes in script used to generate data.
    :param scale: int The scale of the spectrum data.
    :param omega_shift: float Used for generation of x-axis.
    :param dg: float Variation in gamma for liquid modes.
    :param dgs: float Variation in gamma for shell modes.
    :param gamma_amp_factor: float (optional) Scales gammaAmp: `GammaAmp=scale./(1+0.5*dG)./gammaAmpFactor;`
    :param amp_factor: float (optional) Scales Amp0S: `Amp0S=rand(nc.*K,NS)./ampFactor;`
    :param epsilon2: float (optional) Argument that scales white noise: ` D=D + epsilon2.*max(D).
    :return: None
    """

    # Setup data directory
    matlab_script = get_matlab_selection(version)
    directory = os.path.join(DATA_DIR, name)
    try_create_directory(directory)
    check_clear_directory(directory)

    print("Creating generator...")
    
    s3_dir = 'nasa-capstone-data-storage'
    # Uncomment the following line to save data to S3 on generation
    #spectra_generator = S3SpectraGenerator(s3_dir, matlab_script=matlab_script, nc=num_channels, n_max=n_max, n_max_s=n_max_s,
    #                                          scale=scale, omega_shift=omega_shift, dg=dg, dgs=dgs)

    # Comment out the following line to disable local data generation
    spectra_generator = LocalSpectraGenerator(matlab_script=matlab_script, nc=num_channels, n_max=n_max, n_max_s=n_max_s,
                                              scale=scale, omega_shift=omega_shift, dg=dg, dgs=dgs, gamma_amp_factor=gamma_amp_factor,
                                              amp_factor=amp_factor, epsilon2=epsilon2, save_dir=directory)


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
        #save_images(directory, spectra_loader, math.ceil(NUM_EXAMPLE_IMAGES/num_shards))

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

            spectra_generator.save_spectra(spectra_train_json, train_name)
            print(f"    Saved {len(spectra_train_json)} spectra")
            num_train_shards_saved += 1
            num_saved += len(spectra_train_json)

        while len(test_set_buffer) >= shard_size or (shard_i == num_shards - 1 and len(test_set_buffer) > 0):
            print("  Saving testing data...")
            spectra_test_json = test_set_buffer[:shard_size]
            test_set_buffer = test_set_buffer[shard_size:]
            test_name = f'{TEST_DATASET_PREFIX}_{name}.pkl' if shard_size == num_instances else \
                f'{TEST_DATASET_PREFIX}_{name}-p{num_test_shards_saved + 1}.{DATASET_FILE_TYPE}'

            spectra_generator.save_spectra(spectra_test_json, test_name)
            print(f"    Saved {len(spectra_test_json)} spectra")
            num_test_shards_saved += 1
            num_saved += len(spectra_test_json)

    print("\nSaving info...")
    spectra_generator.save_metadata(directory)
 
    print(f"Saved {num_saved} spectra to {directory}.\nDone.")


if __name__ == '__main__':
    main()
