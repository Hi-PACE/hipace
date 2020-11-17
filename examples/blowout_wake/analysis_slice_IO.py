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
F_full_sliced = (F_full[:,F_full.shape[1]//2,:].squeeze() +
                  F_full[:,F_full.shape[1]//2-1,:].squeeze())/2.
ds2 = AMReXDataset('slice_io')
ad = ds2.all_data()
F_slice = ad[field].reshape(ds2.domain_dimensions).v.squeeze()

if do_plot:
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.title('full')
    plt.imshow(F_full_sliced)
    plt.colorbar()
    plt.subplot(132)
    plt.title('slice')
    plt.imshow(F_slice)
    plt.colorbar()
    plt.subplot(133)
    plt.title('difference')
    plt.imshow(F_slice-F_full_sliced)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("image.pdf", bbox_inches='tight')

error = np.max(np.abs(F_slice-F_full_sliced)) / np.max(np.abs(F_full_sliced))

print("F_full.shape", F_full.shape)
print("F_slice.shape", F_slice.shape)
print("error", error)

assert(error < 1.e-14)
