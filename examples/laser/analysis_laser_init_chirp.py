#! /usr/bin/env python3

# Copyright 2024
#
# This file is part of HiPACE++.
#
# Authors: Xingjian Hui
# License: BSD-3-Clause-LBNL

import argparse
import numpy as np
import scipy.constants as scc
from openpmd_viewer.addons import LpaDiagnostics

def get_zeta(Ar, m, w0, L):
    laser_module = np.abs(Ar)
    phi_envelop = np.array(np.arctan2(Ar.imag, Ar.real))
    # unwrap phi_envelop
    phi_envelop = np.unwrap(np.unwrap(phi_envelop, axis=0), axis=1)
    # calculate pphi_pz
    pphi_pz = (np.gradient(phi_envelop,m.z, axis=0)).T * scc.c
    pphi_pzpy = (np.gradient(pphi_pz,m.x, axis=0)).T
    nu = np.sum(pphi_pzpy * laser_module) / np.sum(laser_module)
    a = nu * scc.c**2
    b = -scc.c**2
    c = w0**2 * L**2 * nu / 4
    zeta_roots = np.roots([a, b, c])
    return np.max(zeta_roots)

def get_phi2 (Ar, m, tau):
    # get temporal chirp phi2
    temp_chirp = 0
    summ = 0
    laser_module1 = np.abs(Ar)
    phi_envelop = np.unwrap(np.array(np.arctan2(Ar.imag, Ar.real)), axis=0)
    # calculate pphi_pz
    z_diff = np.diff(m.z)
    pphi_pz = (np.diff(phi_envelop, axis=0)).T/ (z_diff/scc.c)
    pphi_pz2 = ((np.diff(pphi_pz, axis=1)) / (z_diff[:len(z_diff)-1]) / scc.c).T
    for i in range(len(m.z)-2):
        for j in range(len(m.x)-2):
            temp_chirp = temp_chirp + pphi_pz2[i,j] * laser_module1[i,j]
            summ = summ + laser_module1[i,j]
    x = temp_chirp * scc.c**2 / summ
    a = 4 * x
    b = -4
    c = tau**4 * x
    zeta_roots = np.roots([a, b, c])
    return np.max(zeta_roots)

def get_centroids(F, x, z):
    index_array = np.mgrid[0:F.shape[0], 0:F.shape[1]][1]
    centroids = np.sum(index_array * np.abs(F**2), axis=1) / np.sum(np.abs(F**2), axis=1)
    return z[centroids.astype(int)]

def get_beta(F, m, k0):
    z_centroids = get_centroids(F.T, m.x, m.z)
    weight = np.mean(np.abs(F.T)**2, axis = np.ndim(F) - 1)
    derivative = np.gradient(z_centroids) / (m.x[1] - m.x[0])
    return (np.sum(derivative * weight) / np.sum(weight)) / k0 / scc.c

parser = argparse.ArgumentParser(description = 'Verify the chirp initialization')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
parser.add_argument('--chirp_type',
                    dest='chirp_type',
                    default='phi2',
                    help='Type of the initialized chirp')
args = parser.parse_args()

ts = LpaDiagnostics(args.output_dir)

Ar, m = ts.get_field(field='laserEnvelope', iteration=0)
lambda0 = .8e-6          # Laser wavelength
w0 = 30.e-6              # Laser waist
L0 = 5e-6
tau = L0 / scc.c         # Laser duration
k0 = 2 * scc.pi / lambda0
print(get_zeta(Ar, m, w0, L0))
if args.chirp_type == 'phi2':
    phi2 = get_phi2(Ar, m, tau)
    assert(np.abs(phi2 - 2.4e-26) / 2.4e-26 < 1e-2)
elif args.chirp_type == 'zeta':
    zeta = get_zeta(Ar, m, w0, L0)
    assert(np.abs(zeta - 2.4e-19) / 2.4e-19 < 1e-2)
elif args.chirp_type == 'beta':
    beta = get_beta(Ar, m, k0)
    assert(np.abs(beta - 2e-17) / 2e-17 < 1e-2)
