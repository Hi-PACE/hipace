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

def emittance_z(all_data):
    """
    Per-slice emittance: emittance_z(all_data)
    Projected emittance: emittance_z(all_data["average"])
    """
    return np.sqrt(np.abs((all_data["[z^2]"] - all_data["[z]"]**2) \
                  * (all_data["[uz^2]"] - all_data["[uz]"]**2) \
                  - (all_data["[z*uz]"] - all_data["[z]"]*all_data["[uz]"])**2))

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

def temperature_in_ev(all_data, per_slice=True, direction='all'):
    """
    calculated temperature in eV for in-situ diagnostics

    Parameters
    ----------

    all_data: NumPy structured array over timesteps.
        To be obtained via read_file() (see above)
    per_slice: boolean, whether the temperature should be returned per slice or averaged over all slices
    direction: direction along which the temperature should be calculated. Available options:
               'x', 'y', 'z', 'all'

    Returns
    -------

    NumPy structured array over timesteps with the tempature in eV.
    """
    if direction=='all':
        return 1/3.*(temperature_in_ev(all_data, per_slice=per_slice, direction='x') +
                     temperature_in_ev(all_data, per_slice=per_slice, direction='y') +
                     temperature_in_ev(all_data, per_slice=per_slice, direction='z'))
    elif (direction=='x' or direction=='y' or direction=='z'):
        if per_slice:
            if all_data["is_normalized_units"][0]:
                return (constants.m_e * constants.c**2 / constants.e) * normalized_momentum_std(all_data, direction=direction)**2 * np.atleast_2d(all_data["mass"]).T
            else:
                return (constants.c**2 / constants.e) * normalized_momentum_std(all_data, direction=direction)**2 * np.atleast_2d(all_data["mass"]).T
        else:
            if all_data["is_normalized_units"][0]:
                return (constants.m_e * constants.c**2 / constants.e) * normalized_momentum_std(all_data["average"], direction=direction)**2 * all_data["mass"]
            else:
                return (constants.c**2 / constants.e) * normalized_momentum_std(all_data["average"], direction=direction)**2 * all_data["mass"]
    else:
        print("Error, unknown direction, use 'x', 'y', 'z', or 'all'")
        return

def position_std(all_data, direction='x'):
    """
    calculated standard deviation of the position in x,y, or z for in-situ diagnostics

    Parameters
    ----------

    all_data: NumPy structured array over timesteps.
        To be obtained via read_file() (see above)
    direction: direction along which the temperature should be calculated. Available options:
               'x', 'y', 'z'

    Returns
    -------

    NumPy structured array over timesteps.
    """
    if direction=='x':
        var = "[x]"
        var_sq = "[x^2]"
    elif direction=='y':
        var = "[y]"
        var_sq = "[y^2]"
    elif direction=='z':
        var = "[z]"
        var_sq = "[z^2]"
    else:
        print("Error, unknown direction, use 'x', 'y', or 'z'")
        return
    return np.sqrt(np.maximum(all_data[var_sq] - all_data[var]**2,0))

def normalized_momentum_std(all_data, direction='x'):
    """
    calculated standard deviation of the momentum in x,y, or z for in-situ diagnostics

    Parameters
    ----------

    all_data: NumPy structured array over timesteps.
        To be obtained via read_file() (see above)
    direction: direction along which the temperature should be calculated. Available options:
               'x', 'y', 'z'

    Returns
    -------

    NumPy structured array over timesteps.
    """
    if direction=='x':
        var = "[ux]"
        var_sq = "[ux^2]"
    elif direction=='y':
        var = "[uy]"
        var_sq = "[uy^2]"
    elif direction=='z':
        var = "[uz]"
        var_sq = "[uz^2]"
    else:
        print("Error, unknown direction, use 'x', 'y', or 'z'")
        return
    return np.sqrt(np.maximum(all_data[var_sq] - all_data[var]**2,0))

def per_slice_charge(all_data):
    return np.atleast_2d(all_data["charge"]).T * all_data["sum(w)"] * np.atleast_2d(all_data["normalized_density_factor"]).T

def total_charge(all_data):
    return all_data["charge"] * all_data["total"]["sum(w)"] * all_data["normalized_density_factor"]

def z_axis(all_data):
    return (np.linspace(all_data["z_lo"][0], all_data["z_hi"][0], all_data["n_slices"][0]+1)[1:] \
        + np.linspace(all_data["z_lo"][0], all_data["z_hi"][0], all_data["n_slices"][0]+1)[:-1])*0.5


if __name__ == '__main__':
    # To import this file from another script use (change the path):
    #
    # import sys
    # sys.path.append("../hipace/tools/")
    # import read_insitu_diagnostics as diag
    #
    # all_data = diag.read_file("diags/insitu/reduced_beam.*.txt")

    import matplotlib.pyplot as plt

    all_data = read_file("diags/insitu/reduced_beam.*.txt")

    print("Available diagnostics:", all_data.dtype.names)
    print("For instance, all_data['[x]'] returns the per-slice average transverse beam position")

    plt.figure(figsize=(7,7), dpi=150)
    plt.pcolormesh(z_axis(all_data), all_data["time"], energy_spread_eV(all_data, per_slice=True))

    plt.figure(figsize=(7,7), dpi=150)
    plt.plot(all_data["time"], emittance_x(all_data["average"]))
