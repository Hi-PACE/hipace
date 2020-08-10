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

ds = AMReXDataset('plt00001')


do_plot = False

# Radius of the can beam
R = 1

# Define array for transverse coordinate and theory for By and Bx
x = np.linspace(ds.domain_left_edge[0].v, ds.domain_right_edge[0].v, ds.domain_dimensions[0])

By_th = x / 2.
By_th[abs(x)>=R] = R**2/(2*x[abs(x)>R])

y = np.linspace(ds.domain_left_edge[1].v, ds.domain_right_edge[1].v, ds.domain_dimensions[1])
Bx_th = -y / 2.
Bx_th[abs(y)>=R] = -R**2/(2*y[abs(y)>R])

jz_th = - np.ones_like(x)
jz_th[abs(x)>=R] = 0.

# Load Hipace data for By in normalized units
all_data_level_0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
    dims=ds.domain_dimensions)
Bx_sim = all_data_level_0['Bx'].v.squeeze()[ds.domain_dimensions[1]//2,:,ds.domain_dimensions[2]//2]
By_sim = all_data_level_0['By'].v.squeeze()[:,ds.domain_dimensions[1]//2,ds.domain_dimensions[2]//2]
jz_sim = all_data_level_0['jz'].v.squeeze()[:,ds.domain_dimensions[1]//2,ds.domain_dimensions[2]//2]

# Plot simulation result and theory
if do_plot:
    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12,4))

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
print("total relative error jz: " + str(error_jz) + " (tolerance = 0.15)")
assert(error_jz < .15)

error_Bx = np.sum((Bx_sim-Bx_th)**2) / np.sum((Bx_th)**2)
print("total relative error Bx: " + str(error_Bx) + " (tolerance = 0.03)")
assert(error_Bx < .03)

error_By = np.sum((By_sim-By_th)**2) / np.sum((By_th)**2)
print("total relative error By: " + str(error_By) + " (tolerance = 0.03)")
assert(error_By < .03)
