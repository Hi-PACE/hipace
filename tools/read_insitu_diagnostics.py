# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL

import numpy as np
from glob import glob

def properties_table():
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
            "[ga^2]" : 12,
            "Np" : 13}

def read_file(pathname):
    filenames = glob(pathname)

    # change datatype if endianness becomes a problem
    all_file_data = [np.fromfile(file, dtype=np.double) for file in filenames]

    n_steps = len(all_file_data)
    assert(n_steps>0, "no files found")

    # header:
    # 0: header size
    # 1: n_slices
    # 2: time
    # 3: steps
    # 4: charge
    # 5: mass
    # 6: z_lo
    # 7: z_hi

    header_size = all_file_data[0][0]
    n_slices = all_file_data[0][1]
    charge = all_file_data[0][4]
    mass = all_file_data[0][5]
    z_lo = all_file_data[0][6]
    z_hi = all_file_data[0][7]

    for data in all_file_data:
        assert(data[0] == header_size, "inconsistent format")
        assert(data[1] == n_slices, "inconsistent number of slices")
        assert(data[4] == charge, "inconsistent charge")
        assert(data[5] == mass, "inconsistent mass")
        assert(data[6] == z_lo, "inconsistent z_lo")
        assert(data[7] == z_hi, "inconsistent z_hi")

    assert(header_size==8, "unknown format")
    assert(n_slices>0, "no slices")

    all_file_data.sort(key=lambda data : data[2]) # sort by time

    n_properties = len(properties_table())

    all_data = {"n_steps" : n_steps, "n_slices" : n_slices,
                "time" : np.array([data[2] for data in all_file_data]),
                "step" : np.array([data[3] for data in all_file_data]),
                "charge" : charge,
                "mass" : mass,
                "z_lo" : z_lo,
                "z_hi" : z_hi,
                "average" : {},
                "total" : {}}

    # all_data["[x]" etc.] : 2d array over timesteps and slices
    for key, val in properties_table().items():
        all_data[key] = np.array([data[header_size+n_slices*val:header_size+n_slices*(val+1)] for data in all_file_data])

    # all_data["average"]["[x]" etc.] : 1d array over time steps of averaged quantities
    # all_data["total"]["sum(w)" etc.] : 1d array over time steps of total sum of quantities
    for key, val in properties_table().items():
        if key in ["sum(w)", "Np"]:
            all_data["total"][key] = np.sum(all_data[key], axis=1)
        else:
            all_data["average"][key] = np.average(all_data[key], axis=1, weights=all_data["sum(w)"])

    return return_dict

def z_axis(all_data):
    return 0.5*(np.linspace(all_data["z_lo"], all_data["z_hi"], all_data["n_slices"]+1)[1:] \
              + np.linspace(all_data["z_lo"], all_data["z_hi"], all_data["n_slices"]+1)[:-1])

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
