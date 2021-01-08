#! /usr/bin/env python3

# This Python analysis script is part of the code Hipace
#
# It compares a field from a simulation with full IO and from a simulation with only slice IO

import yt ; yt.funcs.mylog.setLevel(50)
import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
from yt.frontends.boxlib.data_structures import AMReXDataset
import math
import argparse

do_plot = False
field = 'Bz'

ds1 = AMReXDataset('full_io')
all_data_level_0 = ds1.covering_grid(level=0,
                                     left_edge=ds1.domain_left_edge,
                                     dims=ds1.domain_dimensions)
F_full = all_data_level_0[field].v.squeeze()
F_full_xz = (F_full[:,F_full.shape[1]//2,:].squeeze() +
             F_full[:,F_full.shape[1]//2-1,:].squeeze())/2.
F_full_yz = (F_full[F_full.shape[0]//2,:,:].squeeze() +
             F_full[F_full.shape[0]//2-1,:,:].squeeze())/2.

ds2 = AMReXDataset('slice_io_xz')
ad = ds2.all_data()
F_slice_xz = ad[field].reshape(ds2.domain_dimensions).v.squeeze()

ds3 = AMReXDataset('slice_io_yz')
ad = ds3.all_data()
F_slice_yz = ad[field].reshape(ds3.domain_dimensions).v.squeeze()

if do_plot:
    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.title('full_xz')
    plt.imshow(F_full_xz)
    plt.colorbar()
    plt.subplot(232)
    plt.title('slice xz')
    plt.imshow(F_slice_xz)
    plt.colorbar()
    plt.subplot(233)
    plt.title('difference')
    plt.imshow(F_slice_xz-F_full_xz)
    plt.colorbar()

    plt.subplot(234)
    plt.title('full_yz')
    plt.imshow(F_full_yz)
    plt.colorbar()
    plt.subplot(235)
    plt.title('slice yz')
    plt.imshow(F_slice_yz)
    plt.colorbar()
    plt.subplot(236)
    plt.title('difference')
    plt.imshow(F_slice_yz-F_full_yz)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("image.pdf", bbox_inches='tight')

error_xz = np.max(np.abs(F_slice_xz-F_full_xz)) / np.max(np.abs(F_full_xz))
error_yz = np.max(np.abs(F_slice_yz-F_full_yz)) / np.max(np.abs(F_full_yz))

print("F_full.shape", F_full.shape)
print("F_slice_xz.shape", F_slice_xz.shape)
print("F_slice_yz.shape", F_slice_yz.shape)
print("error_xz", error_xz)
print("error_yz", error_yz)

assert(error_xz < 1.e-14)
assert(error_yz < 1.e-14)
