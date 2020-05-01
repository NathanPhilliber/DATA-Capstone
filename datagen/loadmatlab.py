from scipy.io import loadmat
from datagen.spectrum import Spectrum


def mat_to_spectra(mat_filepath):
    matdata = loadmat(mat_filepath)

    spectrum = Spectrum(n = matdata["N"][0][0],
                        dm = matdata["Dm"],
                        dg = matdata["Gamma"],
                        dgs = None,
                        peak_locations = None,
                        n_max = matdata["Nmax"][0][0],
                        num_channels = matdata["nc"][0][0],
                        scale = matdata["scale"][0][0],
                        omega_shift = matdata["omega"][0][0],
                        n_max_s = None)

    return spectrum