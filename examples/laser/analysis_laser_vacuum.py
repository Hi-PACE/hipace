#! /usr/bin/env python3

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet
# License: BSD-3-Clause-LBNL


import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
import scipy.constants as scc
import scipy
from openpmd_viewer.addons import LpaDiagnostics

do_plot = False

parser = argparse.ArgumentParser(description='Compare laser propagation in vacuum with theory')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()



ts1 = LpaDiagnostics(args.output_dir)

lam=.8e-6            # Laser wavelength
w0 = 20.e-6          # Laser waist
z0 = .001            # Laser focal position
a0 = 2.              # Laser normalized amplitude
k0 = 2*np.pi/lam     # Laser wavenumber
zr = np.pi*w0**2/lam # Laser Rayleigh length

W0 = np.zeros(len(ts1.iterations))
A0 = np.zeros(len(ts1.iterations))
Z = np.zeros(len(ts1.iterations))
for iteration in ts1.iterations:
    F, m1 = ts1.get_field(iteration=iteration, field='laserEnvelope')
    a_abs = np.abs(F)
    W0[iteration] = 2.*np.sqrt(np.sum(a_abs**2*m1.x**2)/np.sum(a_abs**2))
    A0[iteration] = 2.*np.max(a_abs)
    Z[iteration] = ts1.current_t*scc.c

w0_th = w0*np.sqrt(1+(Z-z0)**2/zr**2)
a0_th = a0 * 2.*w0/w0_th

if do_plot:
    plt.figure()
    plt.subplot(211)
    plt.plot(Z, 1.e6*w0_th, label='theory')
    plt.plot(Z, 1.e6*W0, '+--', label='sim')
    plt.xlabel('z (m)')
    plt.ylabel('width (um)')
    plt.legend()
    plt.subplot(212)
    plt.plot(Z, a0_th, label='theory')
    plt.plot(Z, A0, '+--', label='sim')
    plt.grid()
    plt.xlabel('z (m)')
    plt.ylabel('a0')
    plt.legend()

print("w0_th", w0_th)
print("W0   ", W0)
print("a0_th", a0_th)
print("A0   ", A0)
print("np.std((w0_th-W0)/w0_th)", np.std((w0_th-W0)/w0_th))
print("np.std((a0_th-A0)/a0_th)", np.std((a0_th-A0)/a0_th))

assert(np.std((w0_th-W0)/w0_th) < 2e-3)
assert(np.std((a0_th-A0)/a0_th) < 4e-3)
