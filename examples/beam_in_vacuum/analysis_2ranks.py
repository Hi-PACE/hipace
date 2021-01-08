#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

do_plot = False

ts_ref = OpenPMDTimeSeries('./REF_diags/h5/')
ts = OpenPMDTimeSeries('./diags/h5/')

# ds = AMReXDataset('REF_plt00001')
# ad = ds.all_data()
# dsr = AMReXDataset('plt00001')
# I don't know why, but dsr.all_data() would give wrong results here.
# adr = dsr.covering_grid(level=0, left_edge=dsr.domain_left_edge,
#                         dims=dsr.domain_dimensions)

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

for field in ['ExmBy', 'EypBx', 'Ez', 'Bx', 'By', 'By', 'jz']:
    print('comparing ' + field)
    F = ts_ref.get_field(field=field, iteration=ts.iterations[-1])[0]
    Fr = ts.get_field(field=field, iteration=ts.iterations[-1])[0]
    assert( np.all( F == Fr ) )
