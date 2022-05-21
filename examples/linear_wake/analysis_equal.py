#! /usr/bin/env python3

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, AlexanderSinn
# License: BSD-3-Clause-LBNL


# This script compares multiple fields from two simulations and asserts that they are equal

import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
import math
import argparse
from openpmd_viewer import OpenPMDTimeSeries

parser = argparse.ArgumentParser(
    description='Script to analyze the equality of two simulations')
parser.add_argument('--first',
                    dest='first',
                    required=True)
parser.add_argument('--second',
                    dest='second',
                    required=True)
args = parser.parse_args()

# Replace the string below, to point to your data
tss = OpenPMDTimeSeries(args.first)
tsp = OpenPMDTimeSeries(args.second)

iteration = 0
for field in ['Bx', 'By', 'Ez', 'ExmBy', 'EypBx']:
    Fs, ms = tss.get_field(iteration=iteration, field=field)
    Fp, mp = tsp.get_field(iteration=iteration, field=field)

    error = np.sum((Fp-Fs)**2) / np.sum(Fs**2)

    print(field, 'error = np.sum((Fp-Fs)**2) / np.sum(Fs**2) = ' +str(error))
    assert(error<0.006)
