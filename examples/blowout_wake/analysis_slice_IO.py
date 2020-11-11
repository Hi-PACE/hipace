#! /usr/bin/env python3

# This Python analysis script is part of the code Hipace
#
# It compares the field Ez from a simulation with full IO and from a simulation with only slice IO

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

ds1 = AMReXDataset('full_io')
all_data_level_0 = ds1.covering_grid(level=0,
                                     left_edge=ds1.domain_left_edge,
                                     dims=ds1.domain_dimensions)
Ez_full = all_data_level_0['Ez'].v.squeeze()
Ez_full_sliced = Ez_full[:,Ez_full.shape[1]//2,:].squeeze()

ds2 = AMReXDataset('slice_io')
ad = ds2.all_data()
Ez_slice = ad['Ez'].reshape(ds2.domain_dimensions).v.squeeze()

if do_plot:
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.title('full')
    plt.imshow(Ez_full_sliced)
    plt.colorbar()
    plt.subplot(132)
    plt.title('slice')
    plt.imshow(Ez_slice)
    plt.colorbar()
    plt.subplot(133)
    plt.title('difference')
    plt.imshow(Ez_slice-Ez_full_sliced)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("image.pdf", bbox_inches='tight')

error = np.max(np.abs(Ez_slice-Ez_full_sliced)) / np.max(np.abs(Ez_full_sliced))

print("Ez_full.shape", Ez_full.shape)
print("Ez_slice.shape", Ez_slice.shape)
print("error", error)

assert(np.all(Ez_slice == Ez_full_sliced))
