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

def get_zeta(Ar,m, w0,L):
    # get spatial chirp zeta
    nu = 0
    sum=0
    laser_module1=np.abs(Ar**2)
    z_coord1=np.array(m.z)
    y_coord1=np.array(m.x)
    phi_envelop=np.array(np.arctan2(Ar.imag, Ar.real))
#unwrap phi_envelop
    phi_envelop = np.unwrap(phi_envelop, axis=0)
    phi_envelop = np.unwrap(phi_envelop, axis=1)
    #calculate pphi_pz/
    z_diff = np.diff(z_coord1)
    y_diff = np.diff(y_coord1)
    pphi_pz = (np.diff(phi_envelop, axis=0)).T/ (z_diff)
    pphi_pzpy = ((np.diff(pphi_pz, axis=0)).T/(y_diff))
    for i in range(len(z_coord1)-2):
        for j in range(len(y_coord1)-2):
            nu=nu+pphi_pzpy[i,j]*laser_module1[i,j]
            sum=sum+laser_module1[i,j]
    a = 4 * nu * w0**2 * L**4
    b = -4 * scc.c
    c = nu * w0**2 * L**2
    zeta_solutions = np.roots([a, b, c])
    return np.min(zeta_solutions)

def get_phi2(Ar,m,tau):
    #get temporal chirp phi2
    temp_chirp = 0
    sum=0
    laser_module1=np.abs(Ar)
    z_coord1=np.array(m.z)
    phi_envelop=np.array(np.arctan2(Ar.imag, Ar.real))
#unwrap phi_envelop
    phi_envelop = np.unwrap(phi_envelop, axis=0)
    #calculate pphi_pz/
    z_diff = np.diff(z_coord1)
    pphi_pz = (np.diff(phi_envelop, axis=0)).T/ (z_diff/scc.c)
    pphi_pz2 = ((np.diff(pphi_pz, axis=1))/(z_diff[:len(z_diff)-1])/scc.c).T
    for i in range(len(z_coord1)-2):
        for j in range(len(y_coord1)-2):
            temp_chirp=temp_chirp+pphi_pz2[i,j]*laser_module1[i,j]
            sum=sum+laser_module1[i,j]
    x=temp_chirp*scc.c**2/sum
    a=4*x
    b=-4
    c=tau**4*x
    return np.max([(-b-np.sqrt(b**2-4*a*c))/(2*a),(-b+np.sqrt(b**2-4*a*c))/(2*a)])

parser = argparse.ArgumentParser(description='Verify the chirp initialization')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
parser.add_argument('--chirp_type',
                    dest='chirp_type',
                    default='phi2',
                    help='the type of the initialised chirp')
args = parser.parse_args()



ts = LpaDiagnostics(args.output_dir)

Ar, m = ts.get_field(field='laserEnvelope', iteration=0)


lambda0=.8e-6            # Laser wavelength
w0 = 30.e-6          # Laser waist
L0 = 5e-6
tau = L0 / scc.c     # Laser duration
print(args.chirp_type)
if args.chirp_type == 'phi2' :
    phi2 = get_phi2(Ar, m, tau)
    print(phi2)
    assert(np.abs(phi2-2.4e-26)/2.4e-26 < 2e-2)
elif args.chirp_type == 'zeta' :
    zeta = get_zeta(Ar, m, w0, L0)
    assert(np.abs(zeta-2.4e-26)/2.4e-26 < 2e-2)