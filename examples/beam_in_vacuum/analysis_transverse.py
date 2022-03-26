#! /usr/bin/env python3

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet
# License: BSD-3-Clause-LBNL


# This script compares the transverse field By from a serial and a parallel simulation
# with transverse beam currents and asserts that the results are the same

from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(
    description='Script to analyze the correctness of beam with transverse current')
parser.add_argument('--serial',
                    dest='serial',
                    required=True)
parser.add_argument('--parallel',
                    dest='parallel',
                    required=True)
parser.add_argument('--do-plot',
                    dest='do_plot',
                    action='store_true',
                    default=False,
                    help='Plot figures and save them to file')
args = parser.parse_args()

# Replace the string below, to point to your data
tss = OpenPMDTimeSeries(args.serial)
tsp = OpenPMDTimeSeries(args.parallel)

iteration = 8
field = 'By'
Fs, ms = tss.get_field(iteration=iteration, field=field)
Fp, mp = tsp.get_field(iteration=iteration, field=field)

error = np.sum((Fp-Fs)**2) / np.sum(Fs**2)

print('error = np.sum((Fp-Fs)**2) / np.sum(Fs**2) = ' +str(error))
assert(error<1.e-10)
