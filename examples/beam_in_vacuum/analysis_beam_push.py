#! /usr/bin/env python3

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
import scipy.constants as scc
from openpmd_viewer import OpenPMDTimeSeries

do_plot = True

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

# Numerical parameters of the simulation
field_strength = 0.5
gamma = 1000.
x_std_initial = 1./2.
omega_beta = np.sqrt(field_strength/gamma)

# Load beam particle data
ts = OpenPMDTimeSeries(args.output_dir)
xp, yp, uzp, wp = ts.get_particle(species='beam', iteration=ts.iterations[-1],
                                  var_list=['x', 'y', 'uz', 'w'])

std_theory = x_std_initial * np.abs(np.cos(omega_beta * ts.current_t))
std_sim_x = np.sqrt(np.sum(xp**2*wp)/np.sum(wp))
std_sim_y = np.sqrt(np.sum(yp**2*wp)/np.sum(wp))

if do_plot:
    plt.figure()
    plt.plot(xp, yp, '.')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('image.pdf', bbox_inches='tight')

print("beam width theory      : " + str(std_theory))
print("beam width simulation x: " + str(std_sim_x))
print("beam width simulation y: " + str(std_sim_y))

# Assert sub-permille error
assert((std_sim_x-std_theory)/std_theory < 1.e-3)
assert((std_sim_y-std_theory)/std_theory < 1.e-3)
