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

# Radius of the can beam
R = 10.e-6
# Density of the can beam
dens = 1.e3

# Define array for transverse coordinate and theory for By
x = np.linspace(ds.domain_left_edge[0].v, ds.domain_right_edge[0].v, ds.domain_dimensions[0])
jz = scc.e * scc.c * dens
By_th = scc.mu_0 * jz * x / 2.
By_th[abs(x)>=R] = scc.mu_0 * jz * R**2/(2*x[abs(x)>R])

# Load Hipace data for By
all_data_level_0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
    dims=ds.domain_dimensions)
By_sim = all_data_level_0['By'].v.squeeze()[:,ds.domain_dimensions[1]//2,
    ds.domain_dimensions[2]//2]

# Plot simulation result and theory
matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(8,8))
plt.plot(1.e6*x, By_sim, '+-', label='hipace++')
plt.plot(1.e6*x, By_th, 'k--', label='theory')
plt.grid()
plt.legend()
plt.xlim(-50., 50.)
plt.xlabel('x (um)')
plt.ylabel('By (A/m)')
plt.savefig("beam_in_vacuum.png", bbox_inches="tight")

# Assert small error
error = np.sum((By_sim-By_th)**2) / np.sum((By_th)**2)
print("total relative error: " + str(error))
assert(error < 2./100)
