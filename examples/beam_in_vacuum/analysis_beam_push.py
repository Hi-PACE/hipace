#! /usr/bin/env python3

import yt ; yt.funcs.mylog.setLevel(50)
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.constants as scc
from yt.frontends.boxlib.data_structures import AMReXDataset

import openpmd_viewer

print('############')
print(openpmd_viewer.__file__)

do_plot = True

# Numerical parameters of the simulation
field_strength = 0.5
gamma = 1000.
x_std_initial = 1./2.
omega_beta = np.sqrt(field_strength/gamma)

# Load beam particle data
ds = AMReXDataset('plt00020')
ad = ds.all_data()
xp = ad['beam', 'particle_position_x'].v
yp = ad['beam', 'particle_position_y'].v
uzp = ad['beam', 'particle_uz'].v
wp = ad['beam', 'particle_w'].v

std_theory = x_std_initial * np.abs(np.cos(omega_beta * ds.current_time))
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
