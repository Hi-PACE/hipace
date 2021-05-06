#! /usr/bin/env python3

# This Python analysis script is part of the code Hipace++
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

import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
import math
import argparse
from openpmd_viewer import OpenPMDTimeSeries

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

ts_norm = OpenPMDTimeSeries(args.norm_data)
ts_si = OpenPMDTimeSeries(args.si_data)
ts_si_fixed_weight = OpenPMDTimeSeries(args.si_fixed_weight_data)

elec_density = 2.8239587008591567e23 # [1/m^3]
# calculation of the plasma frequency
omega_p = np.sqrt( elec_density * (scc.e**2)/ (scc.epsilon_0 * scc.m_e));
E_0 = omega_p * scc.m_e * scc.c / scc.e;

kp = omega_p / scc.c  # 1./10.e-6

# Load Hipace++ data for Ez in both normalized and SI units
Ez_along_z_norm, meta_norm = ts_norm.get_field(
    field='Ez', iteration=1, slice_across=['x','y'], slice_relative_position=[0,0])
Ez_along_z_si, meta_si = ts_si.get_field(
    field='Ez', iteration=1, slice_across=['x','y'], slice_relative_position=[0,0])
Ez_along_z_si_fixed_w, meta = ts_si_fixed_weight.get_field(
    field='Ez', iteration=1, slice_across=['x','y'], slice_relative_position=[0,0])
zeta_norm = meta_norm.z
zeta_si = meta_si.z

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
