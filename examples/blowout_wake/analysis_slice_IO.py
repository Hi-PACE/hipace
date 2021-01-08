#! /usr/bin/env python3

# This Python analysis script is part of the code Hipace
#
# It compares a field from a simulation with full IO and from a simulation with only slice IO

import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
import math
import argparse
from openpmd_viewer import OpenPMDTimeSeries

do_plot = False
field = 'Bz'

ts1 = OpenPMDTimeSeries('full_io')
F_full = ts1.get_field(field=field, iteration=ts1.iterations[-1])[0]
F_full_sliced = (F_full[:,F_full.shape[1]//2,:].squeeze() +
                  F_full[:,F_full.shape[1]//2-1,:].squeeze())/2.

ts2 = OpenPMDTimeSeries('slice_io')
F_slice = ts2.get_field(field=field, iteration=ts2.iterations[-1])[0]

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
