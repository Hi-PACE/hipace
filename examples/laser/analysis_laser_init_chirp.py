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

def get_duration(Ar,m):
    weights=np.abs(Ar**2)
    mean_val = np.average(m.z, weights=np.sum(weights,axis=1))
    std = np.sqrt(np.average((m.z - mean_val) ** 2, weights=np.sum(weights,axis=1)))
    return 2*std/scc.c

def get_phi2 (Ar, m):
    # get temporal chirp phi2
    tau = get_duration(Ar,m)
    laser_module1 = np.abs(Ar**2)
    phi_envelop = np.unwrap( np.unwrap(np.array(np.arctan2(Ar.imag, Ar.real)), axis=0), axis=1)
    # calculate pphi_pz
    pphi_pz = np.gradient(phi_envelop, (m.z[1]-m.z[0])/scc.c, axis=0)
    pphi_pz2 = np.gradient(pphi_pz, (m.z[1]-m.z[0])/scc.c, axis=0)
    temp_chirp = np.average(pphi_pz2, weights=laser_module1)       
    x = temp_chirp
    a = 4 * x
    b = -4
    c = tau**4 * x
    return np.max(np.roots([a, b, c]))

def get_centroids(F, x, z):
    index_array = np.mgrid[0:F.shape[0], 0:F.shape[1]][1]
    centroids = np.sum(index_array * np.abs(F**2), axis=1) / np.sum(np.abs(F**2), axis=1)
    return z[centroids.astype(int)]

def temporal2spectral_fft(Ar,m,k0):
    spect=np.fft.ifft(
            Ar, axis=1, norm="backward"
        )
    Nt = len(m.z)
    dt= (m.z[1]-m.z[0])/scc.c
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt) + k0 *scc.c
    return omega,spect

def get_zeta(Ar,m,k0):
    omega,env_spec=temporal2spectral_fft(Ar,m,k0)
    env_spec_abs = np.abs(env_spec**2)
    yda = np.sum(m.x * env_spec_abs, axis=1) / np.sum(env_spec_abs, axis=1)
    derivative_y_zeta = np.gradient(yda, omega)
    weight_y_2d = np.mean(env_spec_abs, axis=1)
    zeta_y = np.average(derivative_y_zeta.T, weights=weight_y_2d)
    return zeta_y
    
def get_beta(Ar,m,k0):
    omega,env_spec=temporal2spectral_fft(Ar,m,k0)
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
print(get_zeta(Ar, m,k0))
if args.chirp_type == 'phi2':
    phi2 = get_phi2(Ar, m)
    assert(np.abs(phi2 - 2.4e-26) / 2.4e-26 < 1e-2)
elif args.chirp_type == 'beta':
    beta = get_beta(Ar, m, k0)
    assert(np.abs(beta - 2e-17) / 2e-17 < 1e-2)
elif args.chirp_type == 'zeta':
    zeta = get_zeta(Ar, m,k0)
    assert(np.abs(zeta - 2.4e-22) / 2.4e-22 < 1e-2)
