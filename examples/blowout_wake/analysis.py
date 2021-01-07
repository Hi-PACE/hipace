#! /usr/bin/env python3

# This Python analysis script is part of the code Hipace
#
# It compares the transverse field By with the theoretical value, plots both
# the simulation result and the theory on the same plot, and asserts that the
# difference is small.
#
# To use it, run the simulation and execute this script with
# > ../../build/bin/hipace inputs_SI
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

import openpmd_viewer

print('############')
print(openpmd_viewer.__file__)

import argparse

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--normalized-data',
                    dest='norm_data',
                    required=True,
                    help='Path to the data of the normalized units run')
parser.add_argument('--si-data',
                    dest='si_data',
                    required=True,
                    help='Path to the data of the SI units run')
parser.add_argument('--si-fixed-weight-data',
                    dest='si_fixed_weight_data',
                    required=True,
                    help='Path to the data of the SI units run with a fixed weight beam')
parser.add_argument('--do-plot',
                    dest='do_plot',
                    action='store_true',
                    default=False,
                    help='Plot figures and save them to file')
args = parser.parse_args()

ds_norm = AMReXDataset(args.norm_data)
ds_si = AMReXDataset(args.si_data)
ds_si_fixed_weight = AMReXDataset(args.si_fixed_weight_data)

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

all_data_level_0_si_fixed_w = ds_si_fixed_weight.covering_grid(level=0, left_edge=ds_si.domain_left_edge,
    dims=ds_si.domain_dimensions)
Ez_along_z_si_fixed_w = all_data_level_0_si_fixed_w['Ez'].v.squeeze()[ds_si.domain_dimensions[0]//2,
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
error_Ez_fixed_weight = np.sum((Ez_along_z_si_fixed_w-Ez_along_z_si)**2) / np.sum((Ez_along_z_si)**2)
print("total relative error Ez for a fixed weight beam to the fixed ppc beam: " + str(error_Ez_fixed_weight) + " (tolerance = 1e-2)")
assert(error_Ez < 1e-10)
assert(error_Ez_fixed_weight < 1e-2)
