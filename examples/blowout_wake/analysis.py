#! /usr/bin/env python3

# This Python analysis script is part of the code Hipace
#
# It compares the transverse field By with the theoretical value, plots both
# the simulation result and the theory on the same plot, and asserts that the
# difference is small.
#
# To use it, run the simulation and execute this script with
# > ../../build/bin/hipace inputs
# > python analysis.py
# Note: the simulation may take some time, as the box size must be high to have
# decent agreement

import yt ; yt.funcs.mylog.setLevel(50)
import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
from yt.frontends.boxlib.data_structures import AMReXDataset
import math

import argparse

def assert_exit(condition):
    try:
        assert(condition)
    except AssertionError:
        sys.exit(1)

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--normalized-units',
                    dest='norm_units',
                    action='store_true',
                    default=False,
                    help='Run the analysis in normalized units')
parser.add_argument('--normalized-data',
                    dest='norm_data',
                    default=False,
                    help='Path to the data of the normalized units run')
parser.add_argument('--si-data',
                    dest='si_data',
                    default=False,
                    help='Path to the data of the SI units run')
parser.add_argument('--do-plot',
                    dest='do_plot',
                    action='store_true',
                    default=False,
                    help='Plot figures and save them to file')
args = parser.parse_args()

if not (args.norm_data):
    print("Error, no path to the normalized data is given")
    sys.exit(1)

if not (args.si_data):
    print("Error, no path to the normalized data is given")
    sys.exit(1)

ds_norm = AMReXDataset(args.norm_data)#AMReXDataset('plt00001')
ds_si = AMReXDataset(args.si_data)

elec_density = 2.8239587008591567e23 # [1/m^3]
# calculation of the plasma frequency
omega_p = np.sqrt( elec_density * (scc.e**2)/ (scc.epsilon_0 * scc.m_e));
E_0 = omega_p * scc.m_e * scc.c / scc.e;

kp = omega_p / scc.c  # 1./10.e-6

# Load Hipace data for Ez in both normalized and SI units
all_data_level_0_norm = ds_norm.covering_grid(level=0, left_edge=ds_norm.domain_left_edge,
    dims=ds_norm.domain_dimensions)
Ez_along_z_norm = all_data_level_0_norm['Ez'].v.squeeze()[ds_norm.domain_dimensions[0]//2,
    ds_norm.domain_dimensions[1]//2, :]

all_data_level_0_si = ds_si.covering_grid(level=0, left_edge=ds_si.domain_left_edge,
    dims=ds_si.domain_dimensions)
Ez_along_z_si = all_data_level_0_si['Ez'].v.squeeze()[ds_si.domain_dimensions[0]//2,
    ds_si.domain_dimensions[1]//2, :]

zeta_norm = np.linspace(ds_norm.domain_left_edge[2].v, ds_norm.domain_right_edge[2].v, ds_norm.domain_dimensions[2])
zeta_si = np.linspace(ds_si.domain_left_edge[2].v, ds_si.domain_right_edge[2].v, ds_si.domain_dimensions[2])

if args.do_plot:
    fig, ax = plt.subplots()
    ax.plot(zeta_norm, Ez_along_z_norm)
    ax.plot(zeta_si*kp, Ez_along_z_si/E_0, linestyle='--')
    ax.set_xlabel('z')
    ax.set_ylabel('Ez/E0')
    plt.savefig('Ez_z.png')

# Assert that the simulation result is close enough to theory
error_Ez = np.sum((Ez_along_z_si/E_0-Ez_along_z_norm)**2) / np.sum((Ez_along_z_norm)**2)
print("total relative error Ez: " + str(error_Ez) + " (tolerance = 1e-10)")
assert_exit(error_Ez < 1e-10)
