#!/usr/bin/env python3

import numpy as np
import scipy.constants as scc
from openpmd_viewer.addons import LpaDiagnostics
import argparse

# Constants
C = scc.c  # Speed of light (precomputing constant)


def get_zeta(Ar, m, w0, L):
    laser_module = np.abs(Ar)
    phi_envelope = np.unwrap(np.angle(Ar), axis=1)

    # Vectorized operations to calculate pphi_pz and pphi_pzpy
    z_diff = np.diff(m.z)
    x_diff = np.diff(m.x)

    pphi_pz = np.diff(phi_envelope, axis=0) / (z_diff[:, None] / C)
    pphi_pzpy = np.diff(pphi_pz, axis=1) / x_diff

    # Vectorized summation
    weighted_sum = np.sum(pphi_pzpy * laser_module[:-2, :-2])
    total_sum = np.sum(laser_module[:-2, :-2])

    nu = weighted_sum / C / total_sum
    a = 4 * nu * w0**2 * L**4
    b = -4 * C
    c = nu * w0**2 * L**2

    zeta_solutions = np.roots([a, b, c])
    return np.min(zeta_solutions)


def get_phi2(Ar, m, tau):
    laser_module = np.abs(Ar)
    phi_envelope = np.unwrap(np.angle(Ar), axis=0)

    z_diff = np.diff(m.z)

    # Vectorized operations to calculate pphi_pz and pphi_pz2
    pphi_pz = np.diff(phi_envelope, axis=0) / (z_diff[:, None] / C)
    pphi_pz2 = np.diff(pphi_pz, axis=1) / (z_diff[:-1][:, None] / C)

    # Vectorized summation
    temp_chirp = np.sum(pphi_pz2 * laser_module[:-2, :-2])
    total_sum = np.sum(laser_module[:-2, :-2])

    x = temp_chirp * C**2 / total_sum
    a = 4 * x
    b = -4
    c = tau**4 * x

    zeta_solutions = np.roots([a, b, c])
    return np.max(zeta_solutions)


def get_centroids(F, x, z):
    # Vectorized computation of centroids
    index_array = np.arange(F.shape[1])[None, :]  # Faster with broadcasting
    centroids = np.sum(index_array * np.abs(F)**2, axis=1) / np.sum(np.abs(F)**2, axis=1)
    return z[centroids.astype(int)]


def get_beta(F, m, k0):
    z_centroids = get_centroids(F.T, m.x, m.z)
    weight = np.mean(np.abs(F.T)**2, axis=np.ndim(F) - 1)

    # Vectorized derivative calculation
    derivative = np.gradient(z_centroids) / (m.x[1] - m.x[0])

    return np.sum(derivative * weight) / np.sum(weight) / k0 / C


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
tau = L0 / C  # Laser duration
k0 = 2 * np.pi / 800e-9
if args.chirp_type == 'phi2':
    phi2 = get_phi2(Ar, m, tau)
    assert np.abs(phi2 - 2.4e-26) / 2.4e-26 < 1e-2
elif args.chirp_type == 'zeta':
    zeta = get_zeta(Ar, m, w0, L0)
    assert np.abs(zeta - 2.4e-19) / 2.4e-19 < 1e-2
elif args.chirp_type == 'beta':
    beta = get_beta(Ar, m, k0)
    assert np.abs(beta - 2e-17) / 2e-17 < 1e-2
