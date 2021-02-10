#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.constants as scc
from openpmd_viewer import OpenPMDTimeSeries

do_plot = True

# Numerical parameters of the simulation
field_strength = 0.5
gamma = 1000.
x_std_initial = 1./2.
omega_beta = np.sqrt(field_strength/gamma)
plasma_density = 1.
kp_inv = scc.c / scc.e * math.sqrt(scc.epsilon_0 * scc.m_e / plasma_density)

# Load beam particle data
ts = OpenPMDTimeSeries('./diags/h5/')
xp, yp, uzp, wp = ts.get_particle(species='beam', iteration=ts.iterations[-1],
                                  var_list=['x', 'y', 'uz', 'w'])

xp /= kp_inv
yp /= kp_inv

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
