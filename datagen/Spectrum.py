from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
