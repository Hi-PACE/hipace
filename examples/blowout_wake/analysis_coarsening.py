#! /usr/bin/env python3

# Copyright 2021 AlexanderSinn
#
# This file is part of HiPACE++.
#
# License: BSD-3-Clause-LBNL


# This Python analysis script is part of the code Hipace
#
# It compares a field from a simulation with full IO and from a simulation with coarse IO

import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
import math
import argparse
from openpmd_viewer import OpenPMDTimeSeries

fields = ['Ez', 'ExmBy', 'EypBx']

ts1 = OpenPMDTimeSeries('fine_io')
ts2 = OpenPMDTimeSeries('coarse_io')

for field in fields:

    F_full = ts1.get_field(field=field, iteration=ts1.iterations[-1])[0]
    F_full_coarse = (F_full[2::5,1::4,1::3] + F_full[2::5,2::4,1::3])/2

    F_coarse = ts2.get_field(field=field, iteration=ts2.iterations[-1])[0]

    print("F_full.shape =", F_full.shape)
    print("F_full_coarse.shape =", F_full_coarse.shape)
    print("F_coarse.shape =", F_coarse.shape)
    error = np.max(np.abs(F_coarse-F_full_coarse)) / np.max(np.abs(F_full_coarse))
    print("error =", error)

    assert(error < 3.e-14)

del ts1
del ts2

