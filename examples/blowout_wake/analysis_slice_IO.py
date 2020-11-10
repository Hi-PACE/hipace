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

ds1 = AMReXDataset('full_io')
all_data_level_0 = ds1.covering_grid(level=0,
                                     left_edge=ds1.domain_left_edge,
                                     dims=ds1.domain_dimensions)
Ez_full = all_data_level_0['Ez'].v.squeeze()

ds2 = AMReXDataset('slice_io')
all_data_level_0 = ds2.covering_grid(level=0, left_edge=ds2.domain_left_edge, dims=ds2.domain_dimensions)
Ez_slice = all_data_level_0['Ez'].v.squeeze()

assert(np.all(Ez_slice[:,Ez_slice.shape[1]//2,:] == Ez_full[:,Ez_full.shape[1]//2,:]))
