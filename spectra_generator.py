"""
This script is a Python3 port of spectra_generator_vlines3.m provided by Dr. Michael Khasin, NASA AMES Research.

Original Author: Dr. Michael Khasin
Python Port Authors: Steven Bradley, Mateo Ibarguen, Nathan Philliber
"""
import numpy as np

def spectra_generator():
    count = 1

    datafile = 'MyTrainingN3K80b'
    max_resonances = 5
    num_columns = 3
    num_channels = 10
    scale = 1
    omega_shift = 10
    while True:
        N = np.floor(np.random.uniform(0, 1) * max_resonances) + 1
        omega = scale * np.random.rand(1, N) + omega_shift
        gamma = scale / max_resonances * (1 + np.dot(1.8, np.subtract(np.random.rand(1, N), 0.5))) / 4

        phase = np.zeros(shape=(1, N))
        amplitude = np.zeros(shape=(1, N))

        phase_0 = 2 * np.pi * np.random.rand(num_channels * num_columns, N)
        amplitude_0 = np.random.rand(num_channels * num_columns, N)

        n = 1000 * omega_shift
        omega_i = 0
        omega_f = 2 * omega_shift + 1
        omega_step = (omega_f - omega_i) / (n - 1)
        Omega = np.arange(omega_i, omega_f, omega_step)

        rng = np.arange(np.floor(n * (1 / 2 - 1 / 2 / omega_shift)), np.floor(n * (1 / 2 + 1 / 2 / omega_shift)))

        for k in range(num_columns):
            offset = 0
            for jj in range((k-1) * (num_channels+1), k * num_channels):
                L = np.zeros(1, n)
                phase[:] = phase_0[jj,:].reshape(N, 1)
                amplitude[:] = amplitude_0[jj, :].reshape(N, 1)

                for i in range(N):
                    L = L + amplitude[i] / 2 * (np.exp(1*i * phase[i]) / (omega[i] + Omega + 1*i * gamma[i]) + np.exp(-1 * i * phase[i]) / (Omega - omega[i] + 1 * i * gamma[i]))

                cF = np.abs(L)
                D = cF ** 2
                D = (D - np.min(D(rng))) / (np.max(D(rng)) - np.min(D(rng)))
                offset = offset + 1
