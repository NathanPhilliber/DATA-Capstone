from utils import *
from datagen.Spectrum import Spectrum
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import re
import json


class SpectraLoader:

    def __init__(self, spectra_json=None, dataset_name=None, subset_prefix=None, eval_now=True):
        self.spectra_json = spectra_json
        self.spectra = None
        self.dataset_name = dataset_name
        self.subset_prefix = subset_prefix

        if eval_now and spectra_json is not None:
            self.spectra = self.load_from_json(spectra_json)

        elif eval_now and dataset_name is not None and subset_prefix is not None:
            self.spectra = self.load_from_dir(dataset_name, subset_prefix)

    def get_data_files(self):
        return SpectraLoader.collect_sharded_files(self.dataset_name, self.subset_prefix)

    def load_from_dir(self, dataset_name, subset_prefix):
        self.dataset_name = dataset_name
        self.subset_prefix = subset_prefix

        files = self.get_data_files()
        return self.load_spectra(files)

    def load_from_json(self, spectra_json):
        self.spectra_json = spectra_json
        return self.load_spectra()

    def load_spectra_json_files(self, datafiles):
        all_data = []
        for filepath in datafiles:
            spectra_json = self.load_spectra_json(filepath)
            all_data.extend(spectra_json)

        return all_data

    def load_spectra_json(self, filepath):
        return pickle.load(open(filepath, 'rb'))

    def load_spectra(self, datafiles=[], del_old=False):
        if self.spectra_json is None:
            self.spectra_json = self.load_spectra_json_files(datafiles)

        if self.spectra is not None:
            del self.spectra
            self.spectra = None

        self.spectra = [Spectrum(**spectrum_json) for spectrum_json in self.spectra_json]
        self.spectra_json = None
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
    def read_dataset_config(dataset_name):
        return json.load(open(os.path.join(DATA_DIR, dataset_name, DATAGEN_CONFIG), "r"))

    @staticmethod
    def collect_sharded_files(dataset_name, subset):
        dataset_path = os.path.join(DATA_DIR, dataset_name)

        files = os.listdir(dataset_path)
        files_filtered = sorted([os.path.join(dataset_path, file)
                          for file in files if re.match(f"{subset}_.+.{DATASET_FILE_TYPE}", file)])

        return files_filtered
