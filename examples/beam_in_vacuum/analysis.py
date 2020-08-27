#! /usr/bin/env python

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
import matplotlib
import numpy as np
import scipy.constants as scc
from yt.frontends.boxlib.data_structures import AMReXDataset
import argparse

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--normalized_units',
                    dest='norm_units',
                    action='store_true',
                    default=False,
                    help='Run the analysis in normalized units')
parser.add_argument('--do_plot',
                    dest='do_plot',
                    action='store_true',
                    default=False,
                    help='Plot figures and save them to file')

args = parser.parse_args()

ds = AMReXDataset('plt00001')

if args.norm_units:
    jz0 = 1.
    mu_0 = 1.
    R = 1.
else:

    # Density of the can beam
    dens = 2.8239587008591567e23 # at this density, 1/kp = 10um, allowing for an easy comparison with normalized units
    # Define array for transverse coordinate and theory for By and Bx
    jz0 = - scc.e * scc.c * dens
    mu_0 = scc.mu_0
    # Radius of the can beam
    R = 10.e-6

x = np.linspace(ds.domain_left_edge[0].v, ds.domain_right_edge[0].v, ds.domain_dimensions[0])
By_th = mu_0 * jz0 * x / 2.
By_th[abs(x)>=R] = mu_0 * jz0 * R**2/(2*x[abs(x)>R])

y = np.linspace(ds.domain_left_edge[1].v, ds.domain_right_edge[1].v, ds.domain_dimensions[1])
Bx_th = -mu_0 * jz0 * y / 2.
Bx_th[abs(y)>=R] = -mu_0 * jz0 * R**2/(2*y[abs(y)>R])

jz_th = np.ones_like(x) * jz0
jz_th[abs(x)>=R] = 0.

# Load Hipace data for By in SI units
all_data_level_0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
    dims=ds.domain_dimensions)
Bx_sim = all_data_level_0['Bx'].v.squeeze()[ds.domain_dimensions[1]//2,:,ds.domain_dimensions[2]//2]
By_sim = all_data_level_0['By'].v.squeeze()[:,ds.domain_dimensions[1]//2,ds.domain_dimensions[2]//2]
jz_sim = all_data_level_0['jz'].v.squeeze()[:,ds.domain_dimensions[1]//2,ds.domain_dimensions[2]//2]

# Plot simulation result and theory
if args.do_plot:
    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12,4))

    if not args.norm_units:
        plt.subplot(131)
        plt.plot(1.e6*y, Bx_sim, '+-', label='hipace++')
        plt.plot(1.e6*y, Bx_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-50., 50.)
        plt.xlabel('y (um)')
        plt.ylabel('Bx (T)')

        plt.subplot(132)
        plt.plot(1.e6*x, By_sim, '+-', label='hipace++')
        plt.plot(1.e6*x, By_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-50., 50.)
        plt.xlabel('x (um)')
        plt.ylabel('By (T)')

        plt.subplot(133)
        plt.plot(1.e6*x, jz_sim, '+-', label='hipace++')
        plt.plot(1.e6*x, jz_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-50., 50.)
        plt.xlabel('x (um)')
        plt.ylabel('jz (A/m2)')
    else:
        plt.subplot(131)
        plt.plot(y, Bx_sim, '+-', label='hipace++')
        plt.plot(y, Bx_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-5., 5.)
        plt.xlabel('kp y')
        plt.ylabel('c Bx / E0')

        plt.subplot(132)
        plt.plot(x, By_sim, '+-', label='hipace++')
        plt.plot(x, By_th, 'k--', label='theory')
        plt.grid()
        plt.legend()
        plt.xlim(-5., 5.)
        plt.xlabel('kp x')
        plt.ylabel('c By / E0')

        plt.subplot(133)
        plt.plot(x, jz_sim, '+-', label='hipace++')
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
assert(error_jz < .1)

error_Bx = np.sum((Bx_sim-Bx_th)**2) / np.sum((Bx_th)**2)
print("total relative error Bx: " + str(error_Bx) + " (tolerance = 0.02)")
assert(error_Bx < .02)

error_By = np.sum((By_sim-By_th)**2) / np.sum((By_th)**2)
print("total relative error By: " + str(error_By) + " (tolerance = 0.02)")
assert(error_By < .02)
