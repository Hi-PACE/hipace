#! /usr/bin/env python3

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL

# This script calculates the sum of jz.
# The beam current and the grid current should cancel each other.

import argparse
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

parser = argparse.ArgumentParser(description='Script to analyze the correctness of the beam in vacuum')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

ts = OpenPMDTimeSeries(args.output_dir)

# Load Hipace data for jz
jz_sim, jz_info = ts.get_field(field='jz', iteration=1)

# Assert that the grid current and the beam current cancel each other
error_jz = np.sum( (jz_sim)**2)
print("sum of jz**2: " + str(error_jz) + " (tolerance = 3e-3)")

assert(error_jz < 3e-3)
