from utils import *
from datagen.spectrum import Spectrum
import matlab.engine
import pickle
import json
import os




def adjust_peak_locations(peak_locations):
    if type(peak_locations) == float:
        return list([peak_locations])
    return [list(p) for p in peak_locations]


def get_num_timesteps(spectrum):
    return spectrum.get_num_timesteps()


class SpectraGenerator:
    DEFAULT_N_MAX = 5.0
    DEFAULT_N_MAX_S = 5.0
    DEFAULT_NC = 10.0
    DEFAULT_SCALE = 1.0
    DEFAULT_OMEGA_SHIFT = 10.0
    DEFAULT_DG = 0.5
    DEFAULT_DGS = 1.8
    DEFAULT_MATLAB = 'spectra_generator'

    def __init__(self, matlab_script=DEFAULT_MATLAB, n_max=DEFAULT_N_MAX, n_max_s=DEFAULT_N_MAX_S, nc=DEFAULT_NC,
                 scale=DEFAULT_SCALE,
                 omega_shift=DEFAULT_OMEGA_SHIFT, dg=DEFAULT_DG, dgs=DEFAULT_DGS):
        self.n_max = float(n_max)
        self.n_max_s = float(n_max_s)
        self.num_channels = float(nc)
        self.scale = float(scale)
        self.omega_shift = float(omega_shift)
        self.dg = dg
        self.dgs = dgs
        self.num_timesteps = None

        os.chdir(GEN_DIR)
        self.matlab_script = matlab_script
        self.engine = matlab.engine.start_matlab()
        self.matlab_mapper = {'spectra_generator_v1.m': self.engine.spectra_generator_v1,
                              'spectra_generator_v2.m': self.engine.spectra_generator_v2}
        self.num_timesteps = None

    def generate_spectrum(self):
        matlab_method = self.matlab_mapper[self.matlab_script]
        n, dm, peak_locations = matlab_method(float(self.n_max), float(self.n_max_s),
                                              float(self.num_channels), float(self.scale),
                                              float(self.omega_shift), float(self.dg),
                                              float(self.dgs), nargout=3)
        dm = [list(d) for d in dm]
        self.num_timesteps = len(dm[0])
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
        # TODO: Fix this.
        num_timesteps = get_num_timesteps(spectra[0])
        self.num_timesteps = num_timesteps
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
        del spectra_generator_dict['matlab_mapper']
        spectra_generator_dict['num_instances'] = num_instances
        spectra_generator_dict['num_timesteps'] = self.num_timesteps

        info_filename = os.path.join(directory, filename)
        with open(info_filename, 'w') as f:
            json.dump(spectra_generator_dict, f, indent=4)
