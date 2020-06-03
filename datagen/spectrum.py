from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


class Spectrum:

    def __init__(self, n, dm, dg, dgs, peak_locations, n_max, num_channels, scale, omega_shift, n_max_s,
                 gamma_amp_factor=None, amp_factor=None, epsilon2=None, n_shell=None, gamma_amp=None, **kwargs):
        self.n = n
        self.dm = dm
        self.peak_locations = peak_locations
        self.n_max = n_max
        self.n_max_s = n_max_s
        self.num_channels = num_channels
        self.scale = scale
        self.omega_shift = omega_shift
        self.num_channels = num_channels
        self.dg = dg
        self.dgs = dgs
        self.gamma_amp_factor = gamma_amp_factor
        self.amp_factor = amp_factor
        self.epsilon2 = epsilon2
        self.n_shell = n_shell
        self.gamma_amp = gamma_amp

    def get_num_timesteps(self):
        return len(self.dm[0])

    def plot_channel(self, channel_number, ax=None):
        if ax is not None:
            sns.lineplot(x=np.linspace(0, 1, len(self.dm[0])), y=self.dm[channel_number], ax=ax)
        else:
            ax = sns.lineplot(x=np.linspace(0, 1, len(self.dm[0])), y=self.dm[channel_number])
        try:
            peak_locs = self.peak_locations[0]
            p = peak_locs[0]
        except:
            peak_locs = [self.peak_locations]

        for peak in peak_locs:
            ax.axvline(peak, 0, 1, c='red', alpha=0.6)
        return plt

    def plot_channels(self, size, num_channels=None):
        assert num_channels <= self.num_channels
        if num_channels is None:
            num_channels = self.num_channels
        n_rows = max(math.ceil((num_channels + 0.5) / 2), 1)
        if size is None:
            size = (n_rows*5, num_channels*3)
        plt.figure(figsize=size)
        for channel in range(int(num_channels)):
            plt.subplot(n_rows, 2, channel + 1)
            self.plot_channel(channel)
        plt.xticks([])
        plt.yticks([])
        return plt

    def plot_save_channels(self, save_dir, size):
        plt = self.plot_channels(size)
        plt.savefig(save_dir)
