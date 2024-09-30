#!/usr/bin/env python3

# Copyright 2024
#
# This file is part of HiPACE++.
#
# Authors: Xingjian Hui
# License: BSD-3-Clause-LBNL

import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
import scipy.constants as scc
import scipy
from openpmd_viewer.addons import LpaDiagnostics


def get_zeta(Ar, m, w0, L):
    nu = 0
    total_sum = 0
    laser_module = np.abs(Ar)
    phi_envelope = np.arctan2(Ar.imag, Ar.real)

    # Unwrap phi_envelope
    phi_envelope = np.unwrap(np.unwrap(phi_envelope, axis=0), axis=1)

    # Calculate pphi_pz
    z_diff = np.diff(m.z)
    x_diff = np.diff(m.x)
    pphi_pz = np.diff(phi_envelope, axis=0).T / (z_diff / scc.c)
    pphi_pzpy = np.diff(pphi_pz, axis=0).T / x_diff

    for i in range(len(m.z) - 2):
        for j in range(len(m.x) - 2):
            nu += pphi_pzpy[i, j] * laser_module[i, j]
            total_sum += laser_module[i, j]

    nu = nu / scc.c / total_sum
    a = 4 * nu * w0**2 * L**4
    b = -4 * scc.c
    c = nu * w0**2 * L**2
    zeta_solutions = np.roots([a, b, c])

    return np.min(zeta_solutions)


def get_phi2(Ar, m, tau):
    # Get temporal chirp phi2
    temp_chirp = 0
    total_sum = 0
    laser_module1 = np.abs(Ar)
    phi_envelope = np.unwrap(np.arctan2(Ar.imag, Ar.real), axis=0)

    # Calculate pphi_pz
    z_diff = np.diff(m.z)
    pphi_pz = np.diff(phi_envelope, axis=0).T / (z_diff / scc.c)
    pphi_pz2 = np.diff(pphi_pz, axis=1) / (z_diff[:len(z_diff) - 1]) / scc.c

    for i in range(len(m.z) - 2):
        for j in range(len(m.x) - 2):
            temp_chirp += pphi_pz2[i, j] * laser_module1[i, j]
            total_sum += laser_module1[i, j]

    x = temp_chirp * scc.c**2 / total_sum
    a = 4 * x
    b = -4
    c = tau**4 * x
    zeta_solutions = np.roots([a, b, c])

    return np.max(zeta_solutions)


def get_centroids(F, x, z):
    index_array = np.mgrid[0:F.shape[0], 0:F.shape[1]][1]
    centroids = np.sum(index_array * np.abs(F**2), axis=1) / np.sum(np.abs(F**2), axis=1)
    return z[centroids.astype(int)]


def get_beta(F, m):
    k0 = 2 * scc.pi / 800e-9
    z_centroids = get_centroids(F.T, m.x, m.z)
    weight = np.mean(np.abs(F.T)**2, axis=np.ndim(F) - 1)
    derivative = np.gradient(z_centroids) / (m.x[1] - m.x[0])

    return (np.sum(derivative * weight) / np.sum(weight)) / k0 / scc.c


parser = argparse.ArgumentParser(description='Verify the chirp initialization')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
parser.add_argument('--chirp_type',
                    dest='chirp_type',
                    default='phi2',
                    help='The type of the initialized chirp')
args = parser.parse_args()

ts = LpaDiagnostics(args.output_dir)

Ar, m = ts.get_field(field='laserEnvelope', iteration=0)
lambda0 = 0.8e-6  # Laser wavelength
w0 = 30.e-6       # Laser waist
L0 = 5e-6
tau = L0 / scc.c  # Laser duration

if args.chirp_type == 'phi2':
    phi2 = get_phi2(Ar, m, tau)
    assert(np.abs(phi2 - 2.4e-26) / 2.4e-26 < 1e-2)
elif args.chirp_type == 'zeta':
    zeta = get_zeta(Ar, m, w0, L0)
    assert(np.abs(zeta - 2.4e-19) / 2.4e-19 < 1e-2)
elif args.chirp_type == 'beta':
    beta = get_beta(Ar, m)
    assert(np.abs(beta - 2e-17) / 2e-17 < 1e-2)
