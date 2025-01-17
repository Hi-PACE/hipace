#! /usr/bin/env python3

#
# This file is part of HiPACE++.
#
# Authors: EyaDammak

import argparse
import numpy as np
import math
from openpmd_viewer import OpenPMDTimeSeries
import statistics
from scipy.constants import e as qe
parser = argparse.ArgumentParser(
    description='Script to analyze the equality of two simulations')
parser.add_argument('--first',
                    dest='first',
                    required=True,
                    help='Path to the directory containing output files')
parser.add_argument('--second',
                    dest='second',
                    required=True,
                    help='Path to the directory containing output files')
args = parser.parse_args()

ts_linear = OpenPMDTimeSeries(args.first)
ts_circular = OpenPMDTimeSeries(args.second)

a0_linear = 0.00885126
a0_circular = 0.00787934

nc = 1.75e27
n0 = nc / 10000

iteration = 0

rho_elec_linear, _ = ts_linear.get_field(field='rho_elec', coord='z', iteration=iteration)
rho_elec_mean_linear = np.mean(rho_elec_linear, axis=(1, 2))
rho_average_linear = statistics.mean(rho_elec_mean_linear[0:10])
fraction_linear = -rho_average_linear / qe / n0

rho_elec_circular, _ = ts_circular.get_field(field='rho_elec', coord='z', iteration=iteration)
rho_elec_mean_circular = np.mean(rho_elec_circular, axis=(1, 2))
rho_average_circular = statistics.mean(rho_elec_mean_circular[0:10]) #average over a thickness in the ionized region
fraction_circular = -rho_average_circular / qe / n0

fraction_warpx_linear = 0.41014984 # result from WarpX simulation
fraction_warpx_circular = 0.502250841 # result from WarpX simulation

relative_diff_linear = np.abs( ( fraction_linear - fraction_warpx_linear ) / fraction_warpx_linear )
relative_diff_circular = np.abs( ( fraction_circular - fraction_warpx_circular ) / fraction_warpx_circular )

tolerance = 0.25
print(f"fraction_warpx_linear = {fraction_warpx_linear}")
print(f"fraction_hipace_linear = {fraction_linear}")
print(f"fraction_warpx_circular = {fraction_warpx_circular}")
print(f"fraction_hipace_circular = {fraction_circular}")

assert ( (relative_diff_linear < tolerance) and (relative_diff_circular < tolerance) ), 'Test laser_ionization did not pass'
