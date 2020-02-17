from utils import *
from datagen.spectrum import Spectrum
import matlab.engine
import pickle
import json
import os


class SpectraGenerator:

    DEFAULT_N_MAX = 5.0
    DEFAULT_NC = 10.0
    DEFAULT_K = 1.0
    DEFAULT_SCALE = 1.0
    DEFAULT_OMEGA_SHIFT = 10.0

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

    def create_spectra_info(self, num_instances, directory=".", filename=DATAGEN_CONFIG):
        spectra_generator_dict = self.__dict__

        del spectra_generator_dict['engine']
        spectra_generator_dict['num_instances'] = num_instances

        info_filename = os.path.join(directory, filename)
        with open(info_filename, 'w') as f:
            json.dump(spectra_generator_dict, f, indent=4)
