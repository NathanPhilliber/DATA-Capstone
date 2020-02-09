import matlab.engine
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import click


DEFAULT_N_MAX = 5.0
DEFAULT_NC = 10.0
DEFAULT_K = 1.0
DEFAULT_SCALE = 1.0
DEFAULT_OMEGA_SHIFT = 10.0
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = '/'.join(ROOT_DIR.split('/')[:-2])
os.chdir('spectra_generator')


def spectra_train_test_splitter(spectra_loader, test_size=0.15, random_seed=42):
    spectra = np.array(spectra_loader.spectra)
    n_peaks = np.array(spectra_loader.get_n())
    spectra_train , spectra_test, _, _ = train_test_split(spectra, n_peaks, stratify=n_peaks,
                                                          test_size=test_size, random_seed=random_seed)
    return spectra_train, spectra_test


def create_spectra_directory(directory):
    try:
        os.mkdir(directory)
    except:
        pass


def create_spectra_info(spectra_generator, num_instances, directory):
    spectra_generor_dict = spectra_generator.__dict__
    del spectra_generor_dict['engine']
    spectra_generor_dict['num_instances'] = num_instances
    info_filename = directory + '/sample_info.json'
    with open(info_filename, 'w') as f:
        json.dump(spectra_generor_dict, f, indent=4)


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
    def save_spectra(spectra_json, filename):
        with open(filename, 'wb') as file_out:
            pickle.dump(spectra_json, file_out)

    def generate_save_spectra(self, n_instances, filename):
        spectra_json = self.generate_spectra_json(n_instances)
        self.save_spectra(spectra_json, filename)


class SpectraLoader:
    def __init__(self, spectra_filename=None, spectra_json=None):
        self.spectra_filename = spectra_filename
        self.spectra_json = spectra_json
        self.spectra = self.load_spectra()

    def load_spectra_json(self):
        spectra_file = open(self.spectra_filename, 'rb')
        spectra_json = pickle.load(spectra_file)
        return spectra_json

    def load_spectra(self):
        if self.spectra_json is None:
            self.spectra_json = self.load_spectra_json()
        spectra = [Spectrum(**spectrum_json) for spectrum_json in self.spectra_json]
        del self.spectra_json
        return spectra

    def get_num_instances(self):
        return len(self.spectra)

    def get_dm(self):
        return [spectrum.dm for spectrum in self.spectra]

    def get_n(self):
        return [spectrum.n for spectrum in self.spectra]

    def get_peak_locations(self):
        return [spectrum.peak_locations for spectrum in self.spectra]


@click.command()
@click.option('--num-instances', default=10000, help='Number of instances to create. ')
@click.option('--name', prompt='Spectra are stored in this directory. ')
def main(num_instances, name):
    directory = ROOT_DIR + f'/data/{name}'
    create_spectra_directory(directory)
    spectra_generator = SpectraGenerator()
    spectra_json = spectra_generator.generate_spectra_json(num_instances)
    spectra_loader = SpectraLoader(spectra_json=spectra_json)
    spectra_train, spectra_test = spectra_train_test_splitter(spectra_loader)
    spectra_train_json = [spectrum.__dict__ for spectrum in spectra_train]
    spectra_test_json = [spectrum.__dict__ for spectrum in spectra_test]
    SpectraGenerator.save_spectra(spectra_train_json, f'{directory}/train_{name}.pkl')
    SpectraGenerator.save_spectra(spectra_test_json, f'{directory}/test_{name}.pkl')
    create_spectra_info(spectra_generator, num_instances, directory)


if __name__ == '__main__':
    main()




