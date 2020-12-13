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

import argparse

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
parser.add_argument('--gaussian-beam',
                    dest='gaussian_beam',
                    action='store_true',
                    default=False,
                    help='Run the analysis on the Gaussian beam')
args = parser.parse_args()

ds = AMReXDataset('plt00001')

if args.norm_units:
    kp = 1.
    ne = 1.
    q_e = 1.
else:
    kp = 1./10.e-6
    ne = scc.c**2 / scc.e**2 * scc.m_e * scc.epsilon_0 * kp**2
    q_e = scc.e

nz = ds.domain_dimensions[2]
# Load Hipace data for rho
all_data_level_0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
    dims=ds.domain_dimensions)

rho_along_z = all_data_level_0['rho'].v.squeeze()[ds.domain_dimensions[0]//2,
    ds.domain_dimensions[1]//2, :]

zeta_array = np.linspace(ds.domain_left_edge[2].v, ds.domain_right_edge[2].v, nz)
dzeta = zeta_array[1]-zeta_array[0]

# generating the array with the beam density
nb_array = np.zeros(nz)
beam_starting_position = 1 / kp
distance_to_start_pos =  ds.domain_right_edge[2].v - beam_starting_position
index_beam_head = np.int(distance_to_start_pos / dzeta)
beam_length = 2 / kp
beam_length_i = np.int(beam_length / dzeta)
if (args.gaussian_beam):
    sigma_z = 1.41 / kp
    peak_density = 0.01*ne
    for i in range( int(nz/2) -1):
        nb_array[int(nz/2)-i ] = peak_density * np.exp(-0.5*((i*dzeta)/sigma_z)**2 )
        nb_array[int(nz/2)+i ] = peak_density * np.exp(-0.5*((i*dzeta)/sigma_z)**2 )
else:
    nb_array[nz-index_beam_head-beam_length_i:nz-index_beam_head] = 0.01 * ne

# calculating the second derivative of the beam density array
nb_dzdz = np.zeros(nz)
for i in range(nz-1):
    nb_dzdz[i] = (nb_array[i-1] -2*nb_array[i] + nb_array[i+1]  )/dzeta**2

# calculating the theoretical plasma density (see Timon Mehrling's thesis page 41)
n_th = np.zeros(nz)
for i in np.arange(nz-1,-1,-1):
    tmp = 0.
    for j in range(nz-i):
        tmp += 1./kp*math.sin(kp*dzeta*(i-(nz-1-j)))*nb_dzdz[nz-1-j]
    n_th[i] = tmp*dzeta + nb_array[i]
rho_th = n_th * q_e

if args.do_plot:
    fig, ax = plt.subplots()
    ax.plot(zeta_array, rho_along_z)
    ax.plot(zeta_array, rho_th, linestyle='--')
    ax.set_xlabel('x')
    ax.set_ylabel('rho')
    plt.savefig('rho_z.png')

# Assert that the simulation result is close enough to theory
error_rho = np.sum((rho_along_z-rho_th)**2) / np.sum((rho_th)**2)
print("total relative error rho: " + str(error_rho) + " (tolerance = 0.016)")
assert(error_rho < .016)
