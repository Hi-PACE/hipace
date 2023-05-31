# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL

import numpy as np
from numpy.lib import recfunctions as rfn
import glob
import json
from scipy import constants

def get_buffer(file):
    with open(file, "rb") as f:
        bytes = f.read()
        datatype_obj = json.JSONDecoder().raw_decode(bytes.decode(errors="replace"))
    return {"buffer" : bytes, "dtype" : np.dtype(datatype_obj[0]), "offset" : datatype_obj[1]}

def read_file(filenames):
    """
    Extract insitu diagnostics into a NumPy structured array

    Parameters
    ----------

    filenames: string
        Path and name of all files containing insitu diagnostics of a single beam.
        Use '*' as a wildcard to read in multiple files eg. "diags/insitu/reduced_beam.*.txt"

    Returns
    -------

    NumPy structured array over timesteps.
    Some components contain subarrays over slices.
    For these weighted averages and totals over slices are also available.
    Use .dtype or .dtype.names to get a list of available components
    """
    return np.sort(rfn.stack_arrays([
        np.frombuffer(**get_buffer(file)) for file in glob.iglob(filenames)
    ], usemask=False, asrecarray=False, autoconvert=True), order="time")

def emittance_x(all_data):
    """
    Per-slice emittance: emittance_x(all_data)
    Projected emittance: emittance_x(all_data["average"])
    """
    return np.sqrt(np.abs((all_data["[x^2]"] - all_data["[x]"]**2) \
                  * (all_data["[ux^2]"] - all_data["[ux]"]**2) \
                  - (all_data["[x*ux]"] - all_data["[x]"]*all_data["[ux]"])**2))

def emittance_y(all_data):
    """
    Per-slice emittance: emittance_x(all_data)
    Projected emittance: emittance_x(all_data["average"])
    """
    return np.sqrt(np.abs((all_data["[y^2]"] - all_data["[y]"]**2) \
                  * (all_data["[uy^2]"] - all_data["[uy]"]**2) \
                  - (all_data["[y*uy]"] - all_data["[y]"]*all_data["[uy]"])**2))

def gamma_mean(all_data):
    return all_data["[ga]"]

def energy_mean_eV(all_data, per_slice=False):
    if per_slice:
        if all_data["is_normalized_units"][0]:
            return (constants.m_e * constants.c**2 / constants.e) * gamma_mean(all_data) * np.atleast_2d(all_data["mass"]).T
        else:
            return (constants.c**2 / constants.e) * gamma_mean(all_data) * np.atleast_2d(all_data["mass"]).T
    else:
        if all_data["is_normalized_units"][0]:
            return (constants.m_e * constants.c**2 / constants.e) * gamma_mean(all_data["average"]) * all_data["mass"]
        else:
            return (constants.c**2 / constants.e) * gamma_mean(all_data["average"]) * all_data["mass"]

def gamma_spread(all_data):
    return np.sqrt(np.maximum(all_data["[ga^2]"] - all_data["[ga]"]**2,0))

def energy_spread_eV(all_data, per_slice=False):
    if per_slice:
        if all_data["is_normalized_units"][0]:
            return (constants.m_e * constants.c**2 / constants.e) * gamma_spread(all_data) * np.atleast_2d(all_data["mass"]).T
        else:
            return (constants.c**2 / constants.e) * gamma_spread(all_data) * np.atleast_2d(all_data["mass"]).T
    else:
        if all_data["is_normalized_units"][0]:
            return (constants.m_e * constants.c**2 / constants.e) * gamma_spread(all_data["average"]) * all_data["mass"]
        else:
            return (constants.c**2 / constants.e) * gamma_spread(all_data["average"]) * all_data["mass"]

def position_mean_x(all_data):
    return all_data["[x]"]

def position_mean_y(all_data):
    return all_data["[y]"]

def position_mean_z(all_data):
    return all_data["[z]"]

def position_std_x(all_data):
    return np.sqrt(np.maximum(all_data["[x^2]"] - all_data["[x]"]**2,0))

def position_std_y(all_data):
    return np.sqrt(np.maximum(all_data["[y^2]"] - all_data["[y]"]**2,0))

def position_std_z(all_data):
    return np.sqrt(np.maximum(all_data["[z^2]"] - all_data["[z]"]**2,0))

def normalized_momentum_mean_x(all_data):
    return all_data["[ux]"]

def normalized_momentum_mean_y(all_data):
    return all_data["[uy]"]

def normalized_momentum_mean_z(all_data):
    return all_data["[uz]"]

def normalized_momentum_std_x(all_data):
    return np.sqrt(np.maximum(all_data["[ux^2]"] - all_data["[ux]"]**2,0))

def normalized_momentum_std_y(all_data):
    return np.sqrt(np.maximum(all_data["[uy^2]"] - all_data["[uy]"]**2,0))

def normalized_momentum_std_z(all_data):
    return np.sqrt(np.maximum(all_data["[uz^2]"] - all_data["[uz]"]**2,0))

def per_slice_charge(all_data):
    return np.atleast_2d(all_data["charge"]).T * all_data["sum(w)"] * np.atleast_2d(all_data["normalized_density_factor"]).T

def total_charge(all_data):
    return all_data["charge"] * all_data["total"]["sum(w)"] * all_data["normalized_density_factor"]

def z_axis(all_data):
    return (np.linspace(all_data["z_lo"][0], all_data["z_hi"][0], all_data["n_slices"][0]+1)[1:] \
        + np.linspace(all_data["z_lo"][0], all_data["z_hi"][0], all_data["n_slices"][0]+1)[:-1])*0.5


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    all_data = read_file("diags/insitu/reduced_beam.*.txt")

    plt.figure(figsize=(7,7), dpi=150)
    plt.imshow(energy_spread(all_data), aspect="auto")

    plt.figure(figsize=(7,7), dpi=150)
    plt.plot(all_data["time"], emittance_x(all_data["average"]))
