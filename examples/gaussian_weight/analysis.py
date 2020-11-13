#!/usr/bin/env python

import yt ; yt.funcs.mylog.setLevel(50)
import matplotlib.pyplot as plt
import numpy as np
from yt.frontends.boxlib.data_structures import AMReXDataset
import scipy.constants as scc

do_plot = False

# Load data with yt
ds = AMReXDataset('plt00000')
ad = ds.all_data()

x_avg =  0.e-6
y_avg = 10.e-6
z_avg = 20.e-6
x_std = 30.e-6
y_std = 40.e-6
z_std = 50.e-6
charge = 1.e-9

# Get particle data into numpy arrays
xp = ad['beam', 'particle_position_x'].v
yp = ad['beam', 'particle_position_y'].v
zp = ad['beam', 'particle_position_z'].v
uzp = ad['beam', 'particle_uz'].v
wp = ad['beam', 'particle_w'].v

if do_plot:
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
    
charge_sim = np.sum(wp) * scc.e

assert(np.abs((charge_sim-charge)/charge) < 1.e-3)
assert(np.abs((np.average(xp)-x_avg)) < 5.e-7)
assert(np.abs((np.average(yp)-y_avg)/y_avg) < .02)
assert(np.abs((np.average(zp)-z_avg)/z_avg) < .02)
assert(np.abs((np.std(xp)-x_std)/x_std) < .02)
assert(np.abs((np.std(yp)-y_std)/y_std) < .02)
assert(np.abs((np.std(zp)-z_std)/z_std) < .02)
