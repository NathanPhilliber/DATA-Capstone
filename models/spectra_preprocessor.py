from utils import *
from datagen.spectra_loader import SpectraLoader
import json
import numpy as np
import random
from keras.utils import to_categorical


class SpectraPreprocessor:
    """
    Class responsible for managing spectra loaders and transforming data for training.
    """

    def __init__(self, dataset_name, num_channels, num_instances, use_generator=False, load_train=True):
        """
        Object constructor for Spectra Preprocessor

        :param dataset_name: string for dataset name
        :param num_channels: int number of channels to use from data
        :param num_instances: number of instances of data to use
        :param use_generator: bool for is to use training generator or not
        :param load_train: bool for if to load data immediately
        """
        if load_train:
            self.train_spectra_loader = SpectraLoader(dataset_name=dataset_name, subset_prefix=TRAIN_DATASET_PREFIX, eval_now=not use_generator)
        self.test_spectra_loader = SpectraLoader(dataset_name=dataset_name, subset_prefix=TEST_DATASET_PREFIX, eval_now=not use_generator)

        self.datagen_config = json.load(open(os.path.join(DATA_DIR, dataset_name, DATAGEN_CONFIG), "r"))
        self.max_nc = self.datagen_config['num_channels']
        self.num_channels = num_channels
        self.num_instances = num_instances
        self.num_test_instances = None

    def get_data(self, loader):
        """
        Return reshaped data from loader.

        :param loader: SpectraLoader
        :return: X matrix, y vector
        """
        # TODO: Correct?
        dm = loader.get_dm()
        dm_reshaped = np.array(dm[:self.num_instances])[:, :self.num_channels, :]
        X = dm_reshaped.reshape(dm_reshaped.shape[0], dm_reshaped.shape[2], dm_reshaped.shape[1])
        y = np.array(loader.get_n())
        y = y.reshape(y.shape[0], 1)
        y_reshaped = to_categorical(y[:self.num_instances, :])[:, 1:]
        del y, dm
        return X, y_reshaped

    def transform(self):
        """
        Transform loaded data.

        :return: train and test sets
        """
        X_train, y_train = self.transform_train()
        X_test, y_test = self.transform_test()
        return X_train, y_train, X_test, y_test

    def transform_train(self):
        """
        Get and transform train data
        :return: X, y
        """
        X_train, y_train = self.get_data(self.train_spectra_loader)
        return X_train, y_train

    def transform_test(self):
        """
        Get and transform test data
        :return: X, y
        """
        X_test, y_test = self.get_data(self.test_spectra_loader)
        return X_test, y_test

    def test_generator(self, batch_size):
        """
        Get test generator
        :param batch_size: size of batch
        :return: test generator
        """
        return self._generator(loader=self.test_spectra_loader, transform_func=self.transform_test,
                               batch_size=batch_size)

    def train_generator(self, batch_size):
        """
        Get train generator
        :param batch_size: size of batch
        :return: train generator
        """
        return self._generator(loader=self.train_spectra_loader, transform_func=self.transform_train,
                               batch_size=batch_size)

    def _generator(self, loader, transform_func, batch_size):
        """
        Loads sharded data from directory in chunks of batch size.

        :param loader: SpectraLoader
        :param transform_func: data transformation function
        :param batch_size: size of batch to use
        :return:
        """
        cur_set_i = 0
        files = loader.get_data_files()
        num_files = len(files)
        spectra_x = None
        spectra_y = None

        while True:
            if cur_set_i >= num_files:
                cur_set_i = 0
                random.shuffle(files)

                print("Reached end of files, reshuffling...")

            loader.load_spectra([files[cur_set_i]], del_old=True)
            cur_set_i += 1
            #dat = self.transform_train(encoded=encoded)
            dat = transform_func()

            if spectra_x is None:
                spectra_x = dat[0]
            else:
                spectra_x = np.concatenate((spectra_x, dat[0]))

            if spectra_y is None:
                spectra_y = dat[1]
            else:
                spectra_y = np.concatenate((spectra_y, dat[1]))

            while len(spectra_x) >= batch_size:
                spectra_batch_x = spectra_x[:batch_size]
                spectra_batch_y = spectra_y[:batch_size]
                spectra_x = spectra_x[batch_size:]
                spectra_y = spectra_y[batch_size:]

                yield spectra_batch_x, spectra_batch_y

    def get_num_test_instances(self):
        """
        Find the number of test instances from data directory.
        TODO: store in dataset config
        :return: int
        """
        if self.num_test_instances is not None:
            return self.num_test_instances

        files = self.test_spectra_loader.get_data_files()

        total = 0
        for file in files:
            print(file)
            self.test_spectra_loader.load_spectra([file], del_old=True)
            total += self.test_spectra_loader.get_num_instances()

        print(f"Found {total} test spectra")

        self.num_test_instances = total
        return total
