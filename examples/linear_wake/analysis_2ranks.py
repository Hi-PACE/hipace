#!/usr/bin/env python

import yt ; yt.funcs.mylog.setLevel(50)
import matplotlib.pyplot as plt
import numpy as np
from yt.frontends.boxlib.data_structures import AMReXDataset

do_plot = True

ds = AMReXDataset('REF_plt00001')
ad = ds.all_data()
dsr = AMReXDataset('plt00001')
# I don't know why, but dsr.all_data() would give wrong results here.
adr = dsr.covering_grid(level=0, left_edge=dsr.domain_left_edge,
                        dims=dsr.domain_dimensions)

if do_plot:

    field = 'jx'
    step = 20
    ms = .1

    plt.figure()

    F = ad[field].v.reshape(ds.domain_dimensions).squeeze()
    xp = ad['beam', 'particle_position_x'].v
    yp = ad['beam', 'particle_position_y'].v
    zp = ad['beam', 'particle_position_z'].v
    uzp = ad['beam', 'particle_uz'].v
    wp = ad['beam', 'particle_w'].v

    plt.subplot(221)
    plt.title('ref xz')
    extent = [ds.domain_left_edge[2], ds.domain_right_edge[2],
              ds.domain_left_edge[0], ds.domain_right_edge[0] ]
    plt.imshow(F[:,F.shape[1]//2,:], extent=extent, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(zp[::step], xp[::step], '.', ms=ms)
    plt.colorbar()
    plt.xlabel('z')
    plt.ylabel('x')

    plt.subplot(222)
    plt.title('ref yz')
    extent = [ds.domain_left_edge[2], ds.domain_right_edge[2],
              ds.domain_left_edge[1], ds.domain_right_edge[1] ]
    plt.imshow(F[F.shape[0]//2,:,:], extent=extent, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(zp[::step], yp[::step], '.', ms=ms)
    plt.colorbar()
    plt.xlabel('z')
    plt.ylabel('y')

    Fr = adr[field].v.reshape(dsr.domain_dimensions).squeeze()
    xp = adr['beam', 'particle_position_x'].v
    yp = adr['beam', 'particle_position_y'].v
    zp = adr['beam', 'particle_position_z'].v
    uzp = adr['beam', 'particle_uz'].v
    wp = adr['beam', 'particle_w'].v

    plt.subplot(223)
    plt.title('xz')
    extent = [dsr.domain_left_edge[2], dsr.domain_right_edge[2],
              dsr.domain_left_edge[0], dsr.domain_right_edge[0] ]
    plt.imshow(Fr[:,Fr.shape[1]//2,:], extent=extent, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(zp[::step], xp[::step], '.', ms=ms)
    plt.colorbar()
    plt.xlabel('z')
    plt.ylabel('x')

    plt.subplot(224)
    plt.title('yz')
    extent = [dsr.domain_left_edge[2], dsr.domain_right_edge[2],
              dsr.domain_left_edge[1], dsr.domain_right_edge[1] ]
    plt.imshow(Fr[Fr.shape[0]//2,:,:], extent=extent, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(zp[::step], yp[::step], '.', ms=ms)
    plt.colorbar()
    plt.xlabel('z')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig('img.pdf', bbox_inches='tight')

for field in ['ExmBy', 'EypBx', 'Ez', 'Bx', 'By', 'By', 'jz']:
    print('comparing ' + field)
    F = ad[field].v.reshape(ds.domain_dimensions).squeeze()
    Fr = adr[field].v.reshape(dsr.domain_dimensions).squeeze()
    assert( np.all( F == Fr ) )

