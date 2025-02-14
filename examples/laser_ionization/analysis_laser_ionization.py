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
import scipy.constants as scc

import sys
sys.path.append("../../tools/")
import read_insitu_diagnostics as diag


parser = argparse.ArgumentParser(
    description='Script to analyze the equality of two simulations')
parser.add_argument('--diags_linear',
                    dest='first',
                    required=True,
                    help='Path to the directory containing output files')
parser.add_argument('--diags_circular',
                    dest='second',
                    required=True,
                    help='Path to the directory containing output files')
parser.add_argument('--insitu_linear',
                    dest='third',
                    required=True,
                    help='Path to the directory containing output files')
parser.add_argument('--insitu_circular',
                    dest='fourth',
                    required=True,
                    help='Path to the directory containing output files')
args = parser.parse_args()

# diagnostics for calculation of the fraction of ionization in linear and circular polarization
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
fraction_linear = -rho_average_linear / scc.e / n0

rho_elec_circular, _ = ts_circular.get_field(field='rho_elec', coord='z', iteration=iteration)
rho_elec_mean_circular = np.mean(rho_elec_circular, axis=(1, 2))
rho_average_circular = statistics.mean(rho_elec_mean_circular[0:10]) #average over a thickness in the ionized region
fraction_circular = -rho_average_circular / scc.e / n0

fraction_warpx_linear = 0.41014984 # result from WarpX simulation
fraction_warpx_circular = 0.502250841 # result from WarpX simulation

relative_diff_linear = np.abs( ( fraction_linear - fraction_warpx_linear ) / fraction_warpx_linear )
relative_diff_circular = np.abs( ( fraction_circular - fraction_warpx_circular ) / fraction_warpx_circular )

tolerance = 0.15
print(f"fraction_warpx_linear = {fraction_warpx_linear}")
print(f"fraction_hipace_linear = {fraction_linear}")
print(f"fraction_warpx_circular = {fraction_warpx_circular}")
print(f"fraction_hipace_circular = {fraction_circular}")

# in-situ diagnostics for calculation of the temperature in all directions in circular polarization
insitu_path_linear = f'./{args.third}/reduced_elec.0000.txt'
all_data_linear = diag.read_file(insitu_path_linear)
Tx2_l = all_data_linear['[ux^2]'][0,0]*scc.m_e*scc.c**2/scc.e
Ty2_l = all_data_linear['[uy^2]'][0,0]*scc.m_e*scc.c**2/scc.e
Tz2_l = all_data_linear['[uz^2]'][0,0]*scc.m_e*scc.c**2/scc.e
temp_eV_linear = 1./3*(Tx2_l+Ty2_l+Tz2_l)

insitu_path_circular = f'./{args.fourth}/reduced_elec.0000.txt'
all_data_circular = diag.read_file(insitu_path_circular)
Tx2_c = all_data_circular['[ux^2]'][0,0]*scc.m_e*scc.c**2/scc.e
Ty2_c = all_data_circular['[uy^2]'][0,0]*scc.m_e*scc.c**2/scc.e
Tz2_c = all_data_circular['[uz^2]'][0,0]*scc.m_e*scc.c**2/scc.e
temp_eV_circular = 1./3*(Tx2_c+Ty2_c+Tz2_c)

temp_eV_warpx_linear = 1.00286009
temp_eV_warpx_circular = 9.68224535

print(f"temp_eV_warpx_linear = {temp_eV_warpx_linear}")
print(f"temp_eV_hipace_linear = {temp_eV_linear}")

print(f"temp_eV_warpx_circular = {temp_eV_warpx_circular}")
print(f"temp_eV_hipace_circular = {temp_eV_circular}")

relative_diff_temp_linear = np.abs( ( temp_eV_linear - temp_eV_warpx_linear ) / temp_eV_warpx_linear )
relative_diff_temp_circular = np.abs( ( temp_eV_circular - temp_eV_warpx_circular ) / temp_eV_warpx_circular )

assert ( (relative_diff_linear < tolerance) and \
         (relative_diff_circular < tolerance) and \
         (relative_diff_temp_linear < tolerance) and \
         (relative_diff_temp_circular < tolerance)), \
         'Test laser_ionization did not pass'
