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

def get_buffer(file):
    with open(file, "rb") as f:
        bytes = f.read()
        datatype_obj = json.JSONDecoder().raw_decode(bytes.decode(errors="replace"))
    return {"buffer" : bytes, "dtype" : np.dtype(datatype_obj[0]), "offset" : datatype_obj[1]}

def read_file(pathname):
    return np.sort(rfn.stack_arrays([
        np.frombuffer(**get_buffer(file)) for file in glob.iglob(pathname)
    ], usemask=False, asrecarray=False, autoconvert=True), order="time")

def z_axis(all_data):
    return (np.linspace(all_data["z_lo"][0], all_data["z_hi"][0], all_data["n_slices"][0]+1)[1:] \
        + np.linspace(all_data["z_lo"][0], all_data["z_hi"][0], all_data["n_slices"][0]+1)[:-1])*0.5

def emittance_x(all_data):
    return np.sqrt(np.abs((all_data["[x^2]"] - all_data["[x]"]**2) \
                  * (all_data["[ux^2]"] - all_data["[ux]"]**2) \
                  - (all_data["[x*ux]"] - all_data["[x]"]*all_data["[ux]"])**2))

def emittance_y(all_data):
    return np.sqrt(np.abs((all_data["[y^2]"] - all_data["[y]"]**2) \
                  * (all_data["[uy^2]"] - all_data["[uy]"]**2) \
                  - (all_data["[y*uy]"] - all_data["[y]"]*all_data["[uy]"])**2))

def energy_spread(all_data):
    return np.sqrt(np.maximum(all_data["[ga^2]"]-all_data["[ga]"]**2,0))

def position_mean_x(all_data):
    return all_data["[x]"]

def position_mean_y(all_data):
    return all_data["[y]"]

def position_std_x(all_data):
    return np.sqrt(np.maximum(all_data["[x^2]"] - all_data["[x]"]**2,0))

def position_std_y(all_data):
    return np.sqrt(np.maximum(all_data["[y^2]"] - all_data["[y]"]**2,0))

def per_slice_charge(all_data):
    return all_data["charge"] * all_data["sum(w)"]

def total_charge(all_data):
    return all_data["charge"] * all_data["total"]["sum(w)"]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    all_data = read_file("diags/insitu/reduced_beam.*.txt")

    plt.figure(figsize=(7,7), dpi=150)
    plt.imshow(energy_spread(all_data), aspect="auto")

    plt.figure(figsize=(7,7), dpi=150)
    plt.plot(all_data["time"], emittance_x(all_data["average"]))
