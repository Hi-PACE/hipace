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

def get_phi2 (Ar, m, tau):
    # get temporal chirp phi2
    temp_chirp = 0
    summ = 0
    laser_module1 = np.abs(Ar**2)
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

def temporal2spectral_fft(Ar,m):
    spect=np.fft.ifft(
            Ar, axis=1, norm="backward"
        )
    Nt = len(m.z)
    dt= (m.z[1]-m.z[0])/scc.c
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt) + k0 *scc.c
    return omega,spect

def get_zeta(Ar,m):
    omega,env_spec=temporal2spectral_fft(Ar,m)
    env_spec_abs = np.abs(env_spec**2)
    
    yda = np.sum(m.x * env_spec_abs, axis=1) / np.sum(env_spec_abs, axis=1)
    derivative_y_zeta = np.gradient(yda, omega)
    weight_y_2d = np.mean(env_spec_abs, axis=1)
    print(derivative_y_zeta.shape)
    print(weight_y_2d.shape)
    zeta_y = np.average(derivative_y_zeta.T, weights=weight_y_2d)
    return zeta_y
    
def get_beta(Ar,m,k0):
    omega,env_spec=temporal2spectral_fft(Ar,m)
    env_spec_abs = np.abs(env_spec**2)
    phi_envelop_abs = np.unwrap(
            np.arctan2(env_spec.imag, env_spec.real), axis=0
        )
    angle_y = np.gradient(phi_envelop_abs, m.x[1]-m.x[0], axis=0) / k0
    derivative_y_beta = np.gradient(angle_y, omega, axis=0)
    beta_y = np.average(derivative_y_beta, weights=env_spec_abs)
    return beta_y 

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
