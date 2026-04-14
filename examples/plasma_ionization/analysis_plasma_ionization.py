#! /usr/bin/env python3

# Copyright 2026
#
# This file is part of HiPACE++.
# Read the output of two simulations with plasma ionization (one with grid ionization,
# the other with particles) and check that the results are close.
#
# Authors: Xingjian Hui, AlexanderSinn, MaxThevenet
# License: BSD-3-Clause-LBNL

import argparse
import numpy as np
import math
from openpmd_viewer import OpenPMDTimeSeries
import scipy.constants as scc

parser = argparse.ArgumentParser(
    description='Script to analyze the equality of two simulations')
parser.add_argument('--diags_grid',
                    dest='first',
                    required=True,
                    help='Path to the directory containing output files')
parser.add_argument('--diags_particle',
                    dest='second',
                    required=True,
                    help='Path to the directory containing output files')
args = parser.parse_args()
ts1 = OpenPMDTimeSeries(args.first)
ts2 = OpenPMDTimeSeries(args.second)
Ar1, m1 = ts1.get_field(field='grid_ionization_w_ion_1', iteration=0)
Ar2, m2 = ts2.get_field(field='n_ion_ionlev_1', iteration=0)
dV1 = m1.dx*m1.dy*m1.dz
dV2 = m2.dx*m2.dy*m2.dz
# Relative error on total number of electrons, normalized by that of grid ionization.
ep = np.abs(np.sum(Ar2)*dV2-np.sum(Ar1)*dV1)/(np.sum(Ar1)*dV1)
tolerance = 0.05
assert ep < tolerance, 'Test grid_ionization did not pass'
