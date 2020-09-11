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
from yt.frontends.boxlib.data_structures import AMReXDataset
import math

import argparse

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--do_plot',
                    dest='do_plot',
                    action='store_true',
                    default=False,
                    help='Plot figures and save them to file')
args = parser.parse_args()

ds = AMReXDataset('plt00001')

nz = ds.domain_dimensions[2]
# Load Hipace data for By
all_data_level_0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
    dims=ds.domain_dimensions)

rho_along_z = all_data_level_0['rho'].v.squeeze()[ds.domain_dimensions[0]//2,
    ds.domain_dimensions[1]//2, :]

jz_along_z = all_data_level_0['jz'].v.squeeze()[ds.domain_dimensions[0]//2,
    ds.domain_dimensions[1]//2, :]

zeta_array = np.linspace(ds.domain_left_edge[2].v, ds.domain_right_edge[2].v, nz) #np.linspace(-7.5,2.5, 401)
dzeta = zeta_array[1]-zeta_array[0]

# generating the array with the beam density
nb_array = np.zeros(nz)
beam_starting_position = 1
distance_to_start_pos =  ds.domain_right_edge[2].v - beam_starting_position
index_beam_head = np.int(distance_to_start_pos / dzeta)
beam_length = 2
beam_length_i = np.int(beam_length / dzeta)
nb_array[nz-index_beam_head-beam_length_i:nz-index_beam_head] = 0.01

# calculating the second derivative of the beam density array
nb_dzdz = np.zeros(nz)
for i in range(nz-1):
    nb_dzdz[i] = (nb_array[i-1] -2*nb_array[i] + nb_array[i+1]  )/dzeta**2

# calculating the theoretical plasma density (see Timon Mehrling's thesis page 41)
n_th = np.zeros(nz)
for i in np.arange(nz-1,-1,-1):
    tmp = 0.
    for j in range(nz-i):
        tmp += math.sin(dzeta*(i-(nz-1-j)))*nb_dzdz[nz-1-j]
    n_th[i] = tmp*dzeta + nb_array[i]

if args.do_plot:
    fig, ax = plt.subplots()
    ax.plot(zeta_array, rho_along_z)
    ax.plot(zeta_array, n_th, linestyle='--')
    ax.set_xlabel('kp x')
    ax.set_ylabel('rho')
    plt.savefig('rho_z.png')


# Assert that the simulation result is close enough to theory
error_rho = np.sum((rho_along_z-n_th)**2) / np.sum((n_th)**2)
print("total relative error rho: " + str(error_rho) + " (tolerance = 0.01)")
assert(error_rho < .01)
