from utils import *
import matlab.engine
#import matplotlib.pyplot as plt
#import seaborn as sns
import pickle
import json
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import click
import math
import re


DEFAULT_N_MAX = 5.0
DEFAULT_NC = 10.0
DEFAULT_K = 1.0
DEFAULT_SCALE = 1.0
DEFAULT_OMEGA_SHIFT = 10.0


class Spectrum:

    def __init__(self, n, dm, peak_locations, n_max, nc, k, scale, omega_shift, num_channels, n_max_s, **kwargs):
        self.n = n
        self.dm = dm
        self.peak_locations = peak_locations
        self.n_max = n_max
        self.n_max_s = n_max_s
        self.nc = nc
        self.k = k
        self.scale = scale
        self.omega_shift = omega_shift
        self.num_channels = num_channels

    def plot_channel(self, channel_number):
        sns.lineplot(x=np.linspace(0, 1, len(self.dm[0])), y=self.dm[channel_number])
        for peak in self.peak_locations[0]:
            plt.axvline(peak, 0, 1, c='red', alpha=0.6)

    def plot_channels(self):
        n_rows = (self.num_channels + 0.5) // 2
        fig = plt.figure(figsize=(30, 35))
        for channel in range(int(self.num_channels)):
            plt.subplot(n_rows, 2, channel + 1)
            self.plot_channel(channel)
        plt.show()


class SpectraGenerator:

    def __init__(self, n_max=DEFAULT_N_MAX, n_max_s=5.0, nc=DEFAULT_NC, k=DEFAULT_K, scale=DEFAULT_SCALE,
                 omega_shift=DEFAULT_OMEGA_SHIFT):
        self.n_max = float(n_max)
        self.n_max_s = float(n_max_s)
        self.nc = float(nc)
        self.k = float(k)
        self.num_channels = float(self.nc * self.k)
        self.scale = float(scale)
        self.omega_shift = float(omega_shift)

        os.chdir(GEN_DIR)
        self.engine = matlab.engine.start_matlab()

    def generate_spectrum(self):

        n, dm, peak_locations = self.engine.spectra_generator_simple(float(self.n_max), float(self.n_max_s),
                                                                     float(self.nc), float(self.k), float(self.scale),
                                                                     float(self.omega_shift), nargout=3)
        dm = [list(d) for d in dm]
        if type(peak_locations) == float:
            peak_locations = list([peak_locations])
        else:
            peak_locations = [list(p) for p in peak_locations]
        spectrum = Spectrum(n=n, dm=dm, peak_locations=peak_locations, **self.__dict__)
        return spectrum

    def generate_spectra(self, n_instances):
        return [self.generate_spectrum() for i in range(n_instances)]

    def generate_spectra_json(self, n_instances):
        spectra = self.generate_spectra(n_instances)
        spectra_json = [spectrum.__dict__ for spectrum in spectra]
        return spectra_json

    @staticmethod
    def save_spectra(spectra_json, filename, save_dir="."):
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'wb') as file_out:
            pickle.dump(spectra_json, file_out)

    def generate_save_spectra(self, n_instances, filename):
        filepath = os.path.join(DATA_DIR, filename)
        spectra_json = self.generate_spectra_json(n_instances)
        self.save_spectra(spectra_json, filepath)

    def create_spectra_info(self, num_instances, directory=".", filename="sample_info.json"):
        spectra_generator_dict = self.__dict__

        del spectra_generator_dict['engine']
        spectra_generator_dict['num_instances'] = num_instances

        info_filename = os.path.join(directory, filename)
        with open(info_filename, 'w') as f:
            json.dump(spectra_generator_dict, f, indent=4)


class SpectraLoader:

    def __init__(self, spectra_json=None, dataset_name=None, subset_prefix=None):
        self.spectra_json = spectra_json
        self.spectra = None

        if spectra_json is not None:
            self.spectra = self.load_from_json(spectra_json)

        elif dataset_name is not None and subset_prefix is not None:
            self.spectra = self.load_from_dir(dataset_name, subset_prefix)

    def load_from_dir(self, dataset_name, subset_prefix):
        files = SpectraLoader.collect_sharded_files(dataset_name, subset_prefix)
        return self.load_spectra(files)

    def load_from_json(self, spectra_json):
        self.spectra_json = spectra_json
        return self.load_spectra()

    def load_spectra_json_files(self, datafiles):
        all_data = []
        for filepath in datafiles:
            spectra_json = pickle.load(open(filepath, 'rb'))
            all_data.extend(spectra_json)

        return all_data

    def load_spectra(self, datafiles=[]):
        if self.spectra_json is None:
            self.spectra_json = self.load_spectra_json_files(datafiles)

        self.spectra = [Spectrum(**spectrum_json) for spectrum_json in self.spectra_json]
        del self.spectra_json
        return self.spectra

    def get_num_instances(self):
        return len(self.spectra)

    def get_dm(self):
        return [spectrum.dm for spectrum in self.spectra]

    def get_n(self):
        return [spectrum.n for spectrum in self.spectra]

    def get_peak_locations(self):
        return [spectrum.peak_locations for spectrum in self.spectra]

    def spectra_train_test_splitter(self, test_size=0.15, random_seed=42):
        spectra = np.array(self.spectra)
        n_peaks = np.array(self.get_n())
        spectra_train, spectra_test, _, _ = train_test_split(spectra, n_peaks, stratify=n_peaks,
                                                             test_size=test_size, random_state=random_seed)
        return spectra_train, spectra_test

    @staticmethod
    def collect_sharded_files(dataset_name, subset):
        dataset_path = os.path.join(DATA_DIR, dataset_name)

        files = os.listdir(dataset_path)
        files_filtered = sorted([os.path.join(dataset_path, file)
                          for file in files if re.match(f"{subset}_.+.{DATASET_FILE_TYPE}", file)])

        return files_filtered


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
