import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


class Spectrum:
    """
    A class responsible for storing data related to a single spectrum.
    """
    def __init__(self, n, dm, dg, dgs, peak_locations, n_max, num_channels, scale, omega_shift, n_max_s,
                 gamma_amp_factor=None, amp_factor=None, epsilon2=None, n_shell=None, gamma_amp=None, **kwargs):
        """

        :param n: int Number of liquid modes in 1 window.
        :param dm: np.array The spectrum data obtained from the matlab script.
        :param dg: float Variation in gamma for liquid modes.
        :param dgs: float Variation in gamma for shell modes.
        :param peak_locations: np.array Locations of liquid modes.
        :param n_max: int The maximum number of possible liquid modes per window in script used to generate data.
        :param num_channels: int The number of channels in spectrum data.
        :param scale: int The scale of the spectrum data.
        :param omega_shift: float Used for generation of x-axis.
        :param n_max_s: int Maximum number of shell modes in script used to generate data.
        :param gamma_amp_factor: float (optional) Scales gammaAmp: `GammaAmp=scale./(1+0.5*dG)./gammaAmpFactor;`
        :param amp_factor: float (optional) Scales Amp0S: `Amp0S=rand(nc.*K,NS)./ampFactor;`
        :param epsilon2: float (optional) Argument that scales white noise: ` D=D + epsilon2.*max(D).*rand(1,omega_res)`
        :param n_shell: int Number of shell modes in this spectrum.
        :param gamma_amp: float Gamma Amp value.
        """
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
        """
        Returns the Number of timesteps in this spectrum object.

        :return: int
        """
        return len(self.dm[0])

    def plot_channel(self, channel_number, ax=None):
        """

        :param channel_number: int Channel number to plot.
        :param ax: ax The axis used to plot channels on.
        :return: Return plt A plot of the channel specified.
        """
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
        """

        :param size: tuple The figure size.
        :param num_channels: int Number of channels to plot.
        :return: plt A plot of the spectrum with the number of channels specified.
        """
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
        """

        :param save_dir: str The directory in which to store the plot of the spectrum.
        :param size: int The size of the plot figure.
        :return: None
        """
        plt = self.plot_channels(size)
        plt.savefig(save_dir)
