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
from openpmd_viewer import OpenPMDTimeSeries
#from lasy.utils.laser_utils import get_beta, get_phi2, get_zeta
#from lasy.profiles import FromOpenPMDProfile
#from lasy.laser import Laser

def get_phi2 (Ar, m):
    # get temporal chirp phi2
    tau = get_duration(Ar,m)
    laser_module1 = np.abs(Ar**2)
    phi_envelop = np.unwrap( np.unwrap(np.arctan(Ar.imag/Ar.real), axis=0), axis=1)
    # calculate pphi_pz
    pphi_pz = np.gradient(phi_envelop, (m.z[1]-m.z[0])/scc.c, axis=0)
    pphi_pz2 = np.gradient(pphi_pz, (m.z[1]-m.z[0])/scc.c, axis=0)
    temp_chirp = np.average(pphi_pz2, weights=laser_module1)
    x = temp_chirp
    a = 4 * x
    b = -4
    c = tau**4 * x
    zeta_roots = np.roots([a, b, c])
    return np.max(zeta_roots)
    
def temporal2spectral_fft(Ar, m, k0):
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
    yda = np.sum(m.y * env_spec_abs, axis=1) / np.sum(env_spec_abs, axis=1)
    derivative_y_zeta = np.gradient(yda, omega)
    weight_y_2d = np.mean(env_spec_abs, axis=1)
    #print(len(derivative_y_zeta))
    zeta_y = np.average(derivative_y_zeta.T, weights=weight_y_2d)
    return zeta_y
    
def get_beta(F, m, k0):
    omega,env_spec=temporal2spectral_fft(F,m,k0)
    phi_envelop_abs = np.unwrap(
        np.array(np.arctan2(env_spec.imag, env_spec.real)), axis=1
    )
    angle_y = np.gradient(phi_envelop_abs, m.y, axis=1) / k0
    dtdb= np.gradient(angle_y, omega, axis=0)
    weight = np.abs(env_spec)**2
    return (np.sum(dtdb * weight) / np.sum(weight))
    
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

#print(args.output_dir)
#profile = FromOpenPMDProfile(path=args.output_dir,iteration=0,pol=[1,0],field='laserEnvelope', is_envelope=True, prefix='openpmd')
#laser = Laser(
#        dim="xyt",
#        lo=(np.min(profile.axes['x']), np.min(profile.axes['y']), np.min(profile.axes['t'])),
#        hi=(np.max(profile.axes['x']), np.max(profile.axes['y']), np.max(profile.axes['t'])),
#        npoints=(511, 255, 500),
#        profile=profile,
#     )
ts=OpenPMDTimeSeries(args.output_dir)
Ar, m = ts.get_field(field='laserEnvelope', iteration=0)
k0 = 2 * scc.pi / 0.6e-6
phi2 = get_phi2(Ar, m)
zeta= get_zeta(Ar, m, k0)
beta = get_beta(Ar, m, k0)

print('phi2 is ')
print(phi2)
print('zeta is ')
print([zeta_x, zeta_y])
print('beta is ')
print([beta_x, beta_y])
np.testing.assert_approx_equal(phi2, 3e-24, significant=2)
np.testing.assert_approx_equal(zeta_y,3e-22, significant=2)
#np.testing.assert_approx_equal(beta_y, 3e-18, significant=2)
