# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL

import numpy as np
from glob import glob

def real_properties_table():
    return {"sum(w)" : 0,
            "[x]" : 1,
            "[x^2]" : 2,
            "[y]" : 3,
            "[y^2]" : 4,
            "[ux]" : 5,
            "[ux^2]" : 6,
            "[uy]" : 7,
            "[uy^2]" : 8,
            "[x*ux]" : 9,
            "[y*uy]" : 10,
            "[ga]" : 11,
            "[ga^2]" : 12}

def int_properties_table():
    return {"np" : 0}

def read_file(pathname):
    filenames = glob(pathname)
    all_data = [np.loadtxt(file) for file in filenames]

    n_steps = len(all_data)

    assert(n_steps>0)

    all_data.sort(key=lambda data : data[0])

    n_int_properties = len(int_properties_table())
    n_real_properties = len(real_properties_table())

    n_slices_list = [(len(x)-2)//(n_int_properties + n_real_properties) for x in all_data]
    n_slices = n_slices_list[0]
    assert(n_slices_list.count(n_slices) == len(n_slices_list))

    assert(n_slices>0)

    return_dict = {"n_slices" : n_slices, "n_steps" : n_steps,
                   "step" : np.array([data[0] for data in all_data]),
                   "time" : np.array([data[1] for data in all_data])}

    for key, val in int_properties_table().items():
        return_dict[key] = np.array([data[2+val:2+n_int_properties*n_slices:n_int_properties] for data in all_data])

    for key, val in real_properties_table().items():
        return_dict[key] = np.array([data[2+n_int_properties*n_slices+val::n_real_properties] for data in all_data])

    return_dict["total_weight"] = np.sum(return_dict["sum(w)"], axis=1)

    return_dict["average"] = {}

    for key, val in int_properties_table().items():
        return_dict["average"][key] = np.sum(return_dict["sum(w)"]*return_dict[key], axis=1)/return_dict["total_weight"]

    for key, val in real_properties_table().items():
        return_dict["average"][key] = np.sum(return_dict["sum(w)"]*return_dict[key], axis=1)/return_dict["total_weight"]

    return return_dict

def z_axis(all_data, z_lo, z_hi):
    return 0.5*(np.linspace(z_lo, z_hi, all_data["n_slices"]+1)[1:] \
              + np.linspace(z_lo, z_hi, all_data["n_slices"]+1)[:-1])

def emittance_x(all_data):
    return np.sqrt(np.maximum((all_data["[x^2]"] - all_data["[x]"]**2) \
                  * (all_data["[ux^2]"] - all_data["[ux]"]**2) \
                  - (all_data["[x*ux]"] - all_data["[x]"]*all_data["[ux]"])**2,0))

def emittance_y(all_data):
    return np.sqrt(np.maximum((all_data["[y^2]"] - all_data["[y]"]**2) \
                  * (all_data["[uy^2]"] - all_data["[uy]"]**2) \
                  - (all_data["[y*uy]"] - all_data["[y]"]*all_data["[uy]"])**2,0))

def energy_spread(all_data):
    return np.sqrt(np.maximum(all_data["[ga^2]"]-all_data["[ga]"]**2,0))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    beam_data = read_file("diags/insitu/reduced_beam.*.txt")

    plt.figure(figsize=(7,7), dpi=150)
    plt.imshow(energy_spread(beam_data) * beam_data["sum(w)"], aspect="auto")

    plt.figure(figsize=(7,7), dpi=150)
    plt.plot(beam_data["time"], emittance_x(beam_data["average"]))
