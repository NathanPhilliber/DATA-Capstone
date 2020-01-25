import matlab.engine
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import os

ENGINE = matlab.engine.start_matlab()
DEFAULT_N_MAX = 5.0
DEFAULT_NC = 10.0
DEFAULT_K = 1.0
DEFAULT_SCALE = 1.0
DEFAULT_OMEGA_SHIFT = 10.0
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = '/'.join(ROOT_DIR.split('/')[:-2])


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
        sns.lineplot(x=np.linspace(0, 1, self.dm[0].shape[0]), y=self.dm[channel_number])
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
    def __init__(self, n_max=DEFAULT_N_MAX, n_max_s=0.0, nc=DEFAULT_NC, k=DEFAULT_K, scale=DEFAULT_SCALE,
                 omega_shift=DEFAULT_OMEGA_SHIFT):
        """

        :param n_max:
        :param nc:
        :param k:
        :param scale:
        :param omega_shift:
        """
        self.n_max = n_max
        self.n_max_s = n_max_s
        self.nc = nc
        self.k = k
        self.num_channels = self.nc * self.k
        self.scale = scale
        self.omega_shift = omega_shift

    def generate_spectrum(self):
        n, dm, peak_locations = ENGINE.spectra_generator_simple(self.n_max, self.n_max_s, self.nc, self.k, self.scale,
                                                                self.omega_shift, nargout=3)
        dm_array, peak_array = np.array(dm), np.array(peak_locations)
        del dm, peak_locations
        spectrum = Spectrum(n=n, dm=dm_array, peak_locations=peak_array, **self.__dict__)
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

