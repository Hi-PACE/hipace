#!/usr/bin/env python3

# Copyright 2020-2021 MaxThevenet, Severin Diederichs
#
# This file is part of HiPACE++.
#
# License: BSD-3-Clause-LBNL


import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as scc
import argparse
from openpmd_viewer import OpenPMDTimeSeries

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--normalized-units',
                    dest='norm_units',
                    action='store_true',
                    default=False,
                    help='Run the analysis in normalized units')
parser.add_argument('--do-plot',
                    dest='do_plot',
                    action='store_true',
                    default=False,
                    help='Plot figures and save them to file')
parser.add_argument('--tilted-beam',
                    dest='tilted_beam',
                    action='store_true',
                    default=False,
                    help='Run the analysis with a tilted beam')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

# Load data with yt
ts = OpenPMDTimeSeries(args.output_dir)

if args.norm_units:
    x_avg = 0.
    y_avg = 1.
    z_avg = 2.
    x_std = 3.
    y_std = 4.
    z_std = 5.
    charge = 1.*3.*4.*5.*(2.*np.pi)**(3/2)/(40./64.)**3
else:
    x_avg =  0.e-6
    y_avg = 10.e-6
    z_avg = 20.e-6
    x_std = 30.e-6
    y_std = 40.e-6
    z_std = 50.e-6
    charge = 1.e-9

if args.tilted_beam:
    y_avg = 1.
    z_avg = 2.
    dx_per_dzeta = 0.1
    dy_per_dzeta = -0.2
    duz_per_uz0_dzeta = 0.01
    uz_avg = 1000.

# only required in the normalized units test
ux_avg = 1.
uy_avg = 2.
ux_std = 3.
uy_std = 4.

# Get particle data into numpy arrays
xp, yp, zp, uxp, uyp, uzp, wp = ts.get_particle(
    species='beam', iteration=ts.iterations[0],
    var_list=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'])

if args.do_plot:
    Hx, bins = np.histogram(xp, weights=wp, range=[-200.e-6, 200.e-6], bins=100)
    Hy, bins = np.histogram(yp, weights=wp, range=[-200.e-6, 200.e-6], bins=100)
    Hz, bins = np.histogram(zp, weights=wp, range=[-200.e-6, 200.e-6], bins=100)
    dbins = bins[1]-bins[0]
    xbins = bins[1:]-dbins/2
    plt.figure()
    plt.plot(1.e6*xbins, Hx, label='x')
    plt.plot(1.e6*xbins, Hy, label='y')
    plt.plot(1.e6*xbins, Hz, label='z')
    plt.xlabel('x (um)')
    plt.ylabel('dQ/dx or dy or dz')
    plt.legend()
    plt.savefig('image.pdf', bbox_inches='tight')

if args.tilted_beam:
    # getting xp and yp at z_avg + 1.
    x_tilt_at_1 = xp[ np.logical_and(z_avg + 0.99 < zp, zp < z_avg + 1.01) ]
    y_tilt_at_1 = yp[ np.logical_and(z_avg + 0.99 < zp, zp < z_avg + 1.01) ]
    uz_at_1 = uzp[ np.logical_and(z_avg + 0.99 < zp, zp < z_avg + 1.01) ]
    x_tilt_error = np.abs(np.average(x_tilt_at_1-dx_per_dzeta)/dx_per_dzeta)
    y_tilt_error = np.abs(np.average(y_tilt_at_1-dy_per_dzeta-y_avg)/dy_per_dzeta)
    uz_error = np.abs(np.average( (uz_at_1 - (uz_avg + 1*uz_avg*duz_per_uz0_dzeta) )/
                                  (uz_avg + 1*uz_avg*duz_per_uz0_dzeta ) ))
    assert(x_tilt_error < 5e-3)
    assert(y_tilt_error < 5e-3)
    assert(uz_error < 5e-4)
else:
    if args.norm_units:
        charge_sim = np.sum(wp)
    else:
        charge_sim = np.sum(wp) * scc.e

    assert(np.abs((charge_sim-charge)/charge) < 1.e-3)
    if args.norm_units:
        assert(np.abs((np.average(xp)-x_avg)) < 1e-12)
        assert(np.abs((np.average(yp)-y_avg)/y_avg) < 1e-4)
        assert(np.average(uxp) < 1e-12)
        assert(np.average(uyp) < 1e-12)
    else:
        assert(np.abs((np.average(xp)-x_avg)) < 5e-7)
        assert(np.abs((np.average(yp)-y_avg)/y_avg) < .03)

    assert( np.abs((np.average(zp)-z_avg)/z_avg) < .035)
    assert(np.abs((np.std(xp)-x_std)/x_std) < .03)
    assert(np.abs((np.std(yp)-y_std)/y_std) < .03)
    assert(np.abs((np.std(zp)-z_std)/z_std) < .03)
