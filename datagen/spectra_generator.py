from utils import *
from datagen.spectrum import Spectrum
import matlab.engine
import pickle
import json
import os
import copy

from abc import abstractmethod, ABC
from s3 import S3


def get_num_timesteps(spectrum):
    """

    :param spectrum:
    :return:
    """
    return spectrum.get_num_timesteps()


class SpectraGenerator(ABC):
    """

    """
    DEFAULT_N_MAX = 5.0
    DEFAULT_N_MAX_S = 5.0
    DEFAULT_NC = 10.0
    DEFAULT_SCALE = 1.0
    DEFAULT_OMEGA_SHIFT = 10.0
    DEFAULT_DG = 0.5
    DEFAULT_DGS = 1.8
    DEFAULT_MATLAB = 'spectra_generator'
    DEFAULT_GAMMA_AMP_FACTOR = 4
    DEFAULT_AMP_FACTOR = 5
    DEFAULT_EPSILON2 = 0.05

    def __init__(self, matlab_script=DEFAULT_MATLAB, n_max=DEFAULT_N_MAX, n_max_s=DEFAULT_N_MAX_S, nc=DEFAULT_NC,
                 scale=DEFAULT_SCALE, omega_shift=DEFAULT_OMEGA_SHIFT, dg=DEFAULT_DG, dgs=DEFAULT_DGS,
                 gamma_amp_factor=DEFAULT_GAMMA_AMP_FACTOR, amp_factor=DEFAULT_AMP_FACTOR, epsilon2=DEFAULT_EPSILON2):
        """

        :param matlab_script: str The matlab script used to generate the data.
        :param n_max: int The maximum number of possible liquid modes per window in script used to generate data.
        :param n_max_s: int Maximum number of shell modes in script used to generate data.
        :param nc: int The number of channels in spectrum data.
        :param scale: int The scale of the spectrum data.
        :param omega_shift: float Used for generation of x-axis.
        :param dg: float Variation in gamma for liquid modes.
        :param dgs: float Variation in gamma for shell modes.
        :param gamma_amp_factor: float (optional) Scales gammaAmp: `GammaAmp=scale./(1+0.5*dG)./gammaAmpFactor;`
        :param amp_factor: float (optional) Scales Amp0S: `Amp0S=rand(nc.*K,NS)./ampFactor;`
        :param epsilon2: float (optional) Argument that scales white noise: ` D=D + epsilon2.*max(D).*rand(1,omega_res)`
        """
        self.n_max = float(n_max)
        self.n_max_s = float(n_max_s)
        self.num_channels = float(nc)
        self.scale = float(scale)
        self.omega_shift = float(omega_shift)
        self.dg = dg
        self.dgs = dgs
        self.gamma_amp_factor = float(gamma_amp_factor)
        self.amp_factor = float(amp_factor)
        self.epsilon2 = float(epsilon2)
        self.num_timesteps = None
        self.num_instances = None
        self.metadata = None

        os.chdir(GEN_DIR)
        self.matlab_script = matlab_script
        self.engine = matlab.engine.start_matlab()
        self.matlab_mapper = {'spectra_generator_v1.m': self.engine.spectra_generator_v1,
                              'spectra_generator_v2.m': self.engine.spectra_generator_v2}

    def generate_spectrum(self):
        """
        Uses the matlab script's function to generate the spectrum data based on the class parameters.
        :return: None
        """
        matlab_method = self.matlab_mapper[self.matlab_script]
        n, dm, peak_locations, omega_res, n_shell, gamma_amp = matlab_method(float(self.n_max), float(self.n_max_s),
                                              float(self.num_channels), float(self.scale),
                                              float(self.omega_shift), float(self.dg),
                                              float(self.dgs), float(self.gamma_amp_factor),
                                              float(self.amp_factor), float(self.epsilon2),
                                              nargout=6)
        dm = [list(d) for d in dm]
        self.num_timesteps = len(dm[0])
        if type(peak_locations) == float:
            peak_locations = list([peak_locations])
        else:
            peak_locations = [list(p) for p in peak_locations]
        spectrum = Spectrum(n=n, dm=dm, peak_locations=peak_locations, n_shell=n_shell, gamma_amp=gamma_amp, **self.__dict__)
        return spectrum

    def generate_spectra(self, n_instances):
        """

        :param n_instances: int The Number of instances to generate.
        :return: A list of length `n_instances` that stores 'spectrum' objects.
        """
        return [self.generate_spectrum() for _ in range(n_instances)]

    def generate_spectra_json(self, n_instances):
        """

        :param n_instances:
        :return:
        """
        spectra = self.generate_spectra(n_instances)
        # TODO: Fix this.
        num_timesteps = get_num_timesteps(spectra[0])
        self.num_timesteps = num_timesteps
        spectra_json = [spectrum.__dict__ for spectrum in spectra]
        #self.update_metadata(n_instances)
        return spectra_json

    @abstractmethod
    def save_spectra(self, spectra_json, filename):
        ...

    def generate_save_spectra(self, n_instances, filename):
        """

        :param n_instances:
        :param filename:
        :return:
        """
        self.num_instances = n_instances
        filepath = os.path.join(DATA_DIR, filename)
        spectra_json = self.generate_spectra_json(n_instances)
        self.save_spectra(spectra_json, filepath)

    def update_metadata(self):
        """

        :return:
        """
        spectra_generator_dict = dict()
        spectra_generator_dict['n_max'] = self.n_max
        spectra_generator_dict['n_max_s'] = self.n_max_s
        spectra_generator_dict['num_channels'] = self.num_channels
        spectra_generator_dict['scale'] = self.scale
        spectra_generator_dict['omega_shift'] = self.omega_shift
        spectra_generator_dict['dg'] = self.dg
        spectra_generator_dict['dgs'] = self.dgs
        spectra_generator_dict['num_timesteps'] = self.num_timesteps
        spectra_generator_dict['num_instances'] = self.num_instances
        spectra_generator_dict['matlab_script'] = self.matlab_script

        spectra_generator_dict['gamma_amp_factor'] = self.gamma_amp_factor
        spectra_generator_dict['amp_factor'] = self.amp_factor
        spectra_generator_dict['epsilon2'] = self.epsilon2

        return spectra_generator_dict

    def save_metadata(self, directory=".", filename=DATAGEN_CONFIG):
        """

        :param directory:
        :param filename:
        :return:
        """
        metadata = self.update_metadata()
        info_filename = os.path.join(directory, filename)
        with open(info_filename, 'w') as f:
            json.dump(metadata, f, indent=4)


class LocalSpectraGenerator(SpectraGenerator):
    """
    Class that generates a spectra and saves it to disk.
    """
    def __init__(self, save_dir, **kwargs):
        super(LocalSpectraGenerator, self).__init__(**kwargs)
        self.save_dir = save_dir

    def save_spectra(self, spectra_json, filename):
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, 'wb') as file_out:
            pickle.dump(spectra_json, file_out)


class S3SpectraGenerator(SpectraGenerator):
    """
    Class that generates spectra and saves the generated data to an S3 bucket.
    """
    def __init__(self, bucket_name, **kwargs):
        super(S3SpectraGenerator, self).__init__(**kwargs)
        self.uploader = S3(bucket_name)

    def save_spectra(self, spectra_json, filename):
        self.uploader.upload_json(json.dumps(spectra_json), self.update_metadata(), filename)

