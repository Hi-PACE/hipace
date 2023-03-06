#! /usr/bin/env python3

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet
# License: BSD-3-Clause-LBNL


# This script compares a field from a simulation with
# full IO and from a simulation with only slice IO

import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
import math
import argparse
from openpmd_viewer import OpenPMDTimeSeries

do_plot = False
field = 'Ez'

ts1 = OpenPMDTimeSeries('full_io')
F_full = ts1.get_field(field=field, iteration=ts1.iterations[-1])[0]
F_full = np.swapaxes(F_full,0,2)
F_full_xz = (F_full[:,F_full.shape[1]//2,:].squeeze() +
             F_full[:,F_full.shape[1]//2-1,:].squeeze())/2.
F_full_yz = (F_full[F_full.shape[0]//2,:,:].squeeze() +
             F_full[F_full.shape[0]//2-1,:,:].squeeze())/2.
F_full_cut_xy = F_full[20:45,:,50].squeeze()
F_full_cut_xz = (F_full[32:49,43,:].squeeze() + F_full[32:49,44,:].squeeze())/2.

ts2 = OpenPMDTimeSeries('slice_io_xz')
F_slice_xz = ts2.get_field(field=field, iteration=ts2.iterations[-1])[0].transpose()

ts3 = OpenPMDTimeSeries('slice_io_yz')
F_slice_yz = ts3.get_field(field=field, iteration=ts3.iterations[-1])[0].transpose()

ts4 = OpenPMDTimeSeries('slice_io_cut_xy')
F_slice_cut_xy = ts4.get_field(field=field, iteration=ts4.iterations[-1])[0].squeeze().transpose()

ts5 = OpenPMDTimeSeries('slice_io_cut_xz')
F_slice_cut_xz = ts5.get_field(field=field, iteration=ts5.iterations[-1])[0].transpose()

if do_plot:
    plt.figure(figsize=(12,16))
    plt.subplot(431)
    plt.title('full_xz')
    plt.imshow(F_full_xz)
    plt.colorbar()
    plt.subplot(432)
    plt.title('slice xz')
    plt.imshow(F_slice_xz)
    plt.colorbar()
    plt.subplot(433)
    plt.title('difference')
    plt.imshow(F_slice_xz-F_full_xz)
    plt.colorbar()

    plt.subplot(434)
    plt.title('full_yz')
    plt.imshow(F_full_yz)
    plt.colorbar()
    plt.subplot(435)
    plt.title('slice yz')
    plt.imshow(F_slice_yz)
    plt.colorbar()
    plt.subplot(436)
    plt.title('difference')
    plt.imshow(F_slice_yz-F_full_yz)
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(437)
    plt.title('full_xy')
    plt.imshow(F_full_cut_xy)
    plt.colorbar()
    plt.subplot(438)
    plt.title('cut slice xy')
    plt.imshow(F_slice_cut_xy)
    plt.colorbar()
    plt.subplot(439)
    plt.title('difference')
    plt.imshow(F_slice_cut_xy-F_full_cut_xy)
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(4, 3, 10)
    plt.title('full_xz')
    plt.imshow(F_full_cut_xz)
    plt.colorbar()
    plt.subplot(4, 3, 11)
    plt.title('cut slice xz')
    plt.imshow(F_slice_cut_xz)
    plt.colorbar()
    plt.subplot(4, 3, 12)
    plt.title('difference')
    plt.imshow(F_slice_cut_xz-F_full_cut_xz)
    plt.colorbar()
    plt.tight_layout()

    plt.savefig("image.pdf", bbox_inches='tight')

error_xz = np.max(np.abs(F_slice_xz-F_full_xz)) / np.max(np.abs(F_full_xz))
error_yz = np.max(np.abs(F_slice_yz-F_full_yz)) / np.max(np.abs(F_full_yz))
error_cut_xy = np.max(np.abs(F_slice_cut_xy-F_full_cut_xy)) / np.max(np.abs(F_full_cut_xy))
error_cut_xz = np.max(np.abs(F_slice_cut_xz-F_full_cut_xz)) / np.max(np.abs(F_full_cut_xz))

print("F_full.shape", F_full.shape)
print("F_slice_xz.shape", F_slice_xz.shape)
print("F_slice_yz.shape", F_slice_yz.shape)
print("F_full_cut_xy.shape", F_full_cut_xy.shape)
print("F_full_cut_xz.shape", F_full_cut_xz.shape)
print("error_xz", error_xz)
print("error_yz", error_yz)
print("error_cut_xy", error_cut_xy)
print("error_cut_xz", error_cut_xz)

assert(error_xz < 3.e-14)
assert(error_yz < 3.e-14)
assert(error_cut_xy < 3.e-14)
assert(error_cut_xz < 3.e-14)
