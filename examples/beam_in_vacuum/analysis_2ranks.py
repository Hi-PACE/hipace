#!/usr/bin/env python3

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


import matplotlib.pyplot as plt
import numpy as np
import argparse
from openpmd_viewer import OpenPMDTimeSeries

do_plot = False

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

ts_ref = OpenPMDTimeSeries('./REF_diags/hdf5/')
ts = OpenPMDTimeSeries(args.output_dir)

if do_plot:

    field = 'Bx'
    step = 20
    ms = .1

    plt.figure()

    F, meta = ts_ref.get_field(field=field, iteration=ts_ref.iterations[-1])

    xp, yp, zp, uzp, wp = ts_ref.get_particle(species='beam', iteration=ts_ref.iterations[-1],
                                          var_list=['x', 'y', 'z', 'uz', 'w'])

    plt.subplot(221)
    plt.title('ref xz')
    extent = [meta.zmin, meta.zmax, meta.xmin, meta.xmax]
    plt.imshow(F[:,F.shape[1]//2,:], extent=extent, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(zp[::step], xp[::step], '.', ms=ms)
    plt.colorbar()
    plt.xlabel('z')
    plt.ylabel('x')

    plt.subplot(222)
    plt.title('ref yz')
    plt.imshow(F[F.shape[0]//2,:,:], extent=extent, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(zp[::step], yp[::step], '.', ms=ms)
    plt.colorbar()
    plt.xlabel('z')
    plt.ylabel('y')

    Fr, meta_ref = ts.get_field(field=field, iteration=ts.iterations[-1])
    xp, yp, zp, uzp, wp = ts_ref.get_particle(species='beam', iteration=ts_ref.iterations[-1],
                                              var_list=['x', 'y', 'z', 'uz', 'w'])

    plt.subplot(223)
    plt.title('xz')
    extent = [meta.zmin, meta.zmax, meta.xmin, meta.xmax]
    plt.imshow(Fr[:,Fr.shape[1]//2,:], extent=extent, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(zp[::step], xp[::step], '.', ms=ms)
    plt.colorbar()
    plt.xlabel('z')
    plt.ylabel('x')

    plt.subplot(224)
    plt.title('yz')
    plt.imshow(Fr[Fr.shape[0]//2,:,:], extent=extent, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(zp[::step], yp[::step], '.', ms=ms)
    plt.colorbar()
    plt.xlabel('z')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig('img.pdf', bbox_inches='tight')

for field in ['ExmBy', 'EypBx', 'Ez', 'Bx', 'By', 'By', 'jz_beam']:
    print('comparing ' + field)
    F = ts_ref.get_field(field=field, iteration=ts.iterations[-1])[0]
    Fr = ts.get_field(field=field, iteration=ts.iterations[-1])[0]
    assert( np.all( F == Fr ) )
