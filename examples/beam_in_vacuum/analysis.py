#! /usr/bin/env python3

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


# This script compares the transverse field By with the theoretical value, plots both
# the simulation result and the theory on the same plot, and asserts that the
# difference is small.
#
# To use it, run the simulation and execute this script with
# > ../../build/bin/hipace inputs_SI
# > python analysis.py
# Note: the simulation may take some time, as the box size must be high to have
# decent agreement

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.constants as scc
import argparse
import sys
from openpmd_viewer import OpenPMDTimeSeries

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--normalized-units',
                    dest='norm_units',
                    action='store_true',
                    default=False,
                    help='Run the analysis in normalized units')
parser.add_argument('--do-plot',
                    dest='do_plot',
                    action='store_true',
                    default=False,
                    help='Plot figures and save them to file')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

ts = OpenPMDTimeSeries(args.output_dir)

if args.norm_units:
    c = 1.
    jz0 = -1.
    rho0 = -1.
    mu_0 = 1.
    eps_0 = 1.
    R = 1.
else:
    # Density of the can beam
    dens = 2.8239587008591567e23 # at this density, 1/kp = 10um, allowing for an easy comparison with normalized units
    # Define array for transverse coordinate and theory for By and Bx
    jz0 = - scc.e * scc.c * dens
    rho0 = - scc.e * dens
    c = scc.c
    mu_0 = scc.mu_0
    eps_0 = scc.epsilon_0
    # Radius of the can beam
    R = 10.e-6

# Load HiPACE++ data for By in SI units
Bx_sim, Bx_meta = ts.get_field(field='Bx', iteration=0, slice_across=['x','z'], slice_relative_position=[0,0])
By_sim, By_meta = ts.get_field(field='By', iteration=0, slice_across=['y','z'], slice_relative_position=[0,0])
jz_sim = ts.get_field(field='jz_beam', iteration=0, slice_across=['y','z'], slice_relative_position=[0,0])[0]
rho_sim = ts.get_field(field='rho', iteration=0, slice_across=['y','z'], slice_relative_position=[0,0])[0]
Ex_sim = ts.get_field(field='ExmBy', iteration=0, slice_across=['y','z'], slice_relative_position=[0,0])[0] + c*By_sim
Ey_sim = ts.get_field(field='EypBx', iteration=0, slice_across=['x','z'], slice_relative_position=[0,0])[0] - c*Bx_sim
y = Bx_meta.y
x = By_meta.x

By_th = mu_0 * jz0 * x / 2.
By_th[abs(x)>=R] = mu_0 * jz0 * R**2/(2*x[abs(x)>R])
Ex_th = rho0 / eps_0 * x / 2.
Ex_th[abs(x)>=R] = rho0 / eps_0 * R**2/(2*x[abs(x)>R])

Bx_th = -mu_0 * jz0 * y / 2.
Bx_th[abs(y)>=R] = -mu_0 * jz0 * R**2/(2*y[abs(y)>R])
Ey_th = rho0 / eps_0 * y / 2.
Ey_th[abs(y)>=R] = rho0 / eps_0 * R**2/(2*y[abs(y)>R])

jz_th = np.ones_like(x) * jz0
jz_th[abs(x)>=R] = 0.
rho_th = np.ones_like(x) * rho0
rho_th[abs(x)>=R] = 0.

# Plot simulation result and theory
if args.do_plot:
    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12,4))

    if not args.norm_units:
        plt.subplot(131)
        plt.plot(1.e6*y, Bx_sim, '+-', label='HiPACE++')
        plt.plot(1.e6*y, Bx_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-50., 50.)
        plt.xlabel('y (um)')
        plt.ylabel('Bx (T)')

        plt.subplot(132)
        plt.plot(1.e6*x, By_sim, '+-', label='HiPACE++')
        plt.plot(1.e6*x, By_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-50., 50.)
        plt.xlabel('x (um)')
        plt.ylabel('By (T)')

        plt.subplot(133)
        plt.plot(1.e6*x, jz_sim, '+-', label='HiPACE++')
        plt.plot(1.e6*x, jz_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-50., 50.)
        plt.xlabel('x (um)')
        plt.ylabel('jz (A/m2)')
    else:
        plt.subplot(131)
        plt.plot(y, Bx_sim, '+-', label='HiPACE++')
        plt.plot(y, Bx_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-5., 5.)
        plt.xlabel('kp y')
        plt.ylabel('c Bx / E0')

        plt.subplot(132)
        plt.plot(x, By_sim, '+-', label='HiPACE++')
        plt.plot(x, By_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-5., 5.)
        plt.xlabel('kp x')
        plt.ylabel('c By / E0')

        plt.subplot(133)
        plt.plot(x, jz_sim, '+-', label='HiPACE++')
        plt.plot(x, jz_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-5., 5.)
        plt.xlabel('kp x')
        plt.ylabel('jz /IA')

    plt.tight_layout()

    plt.savefig("beam_in_vacuum.png", bbox_inches="tight")

# Assert that the simulation result is close enough to theory
error_jz = np.sum((jz_sim-jz_th)**2) / np.sum((jz_th)**2)
print("total relative error jz: " + str(error_jz) + " (tolerance = 0.1)")

error_Bx = np.sum((Bx_sim-Bx_th)**2) / np.sum((Bx_th)**2)
print("total relative error Bx: " + str(error_Bx) + " (tolerance = 0.005)")

error_By = np.sum((By_sim-By_th)**2) / np.sum((By_th)**2)
print("total relative error By: " + str(error_By) + " (tolerance = 0.015)")

error_Ex = np.sum((Ex_sim-Ex_th)**2) / np.sum((Ex_th)**2)
print("total relative error Ex: " + str(error_Ex) + " (tolerance = 0.015)")

error_Ey = np.sum((Ey_sim-Ey_th)**2) / np.sum((Ey_th)**2)
print("total relative error Ey: " + str(error_Ey) + " (tolerance = 0.005)")

assert(error_jz < .1)
assert(error_Bx < .005)
assert(error_By < .015)
assert(error_Ex < .015)
assert(error_Ey < .005)
