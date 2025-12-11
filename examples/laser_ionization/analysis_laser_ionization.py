#! /usr/bin/env python3

#
# This file is part of HiPACE++.
#
# Authors: EyaDammak

import argparse
import numpy as np
import math
from openpmd_viewer import OpenPMDTimeSeries
import scipy.constants as scc
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
izmax = 10

rho_elec_linear, _ = ts_linear.get_field(field='rho_elec', coord='z', iteration=iteration)
rho_elec_mean_linear = np.mean(rho_elec_linear, axis=(1, 2))
rho_average_linear = np.mean(rho_elec_mean_linear[0:izmax])
fraction_linear = -rho_average_linear / scc.e / n0

rho_elec_circular, _ = ts_circular.get_field(field='rho_elec', coord='z', iteration=iteration)
rho_elec_mean_circular = np.mean(rho_elec_circular, axis=(1, 2))
rho_average_circular = np.mean(rho_elec_mean_circular[0:izmax]) #average over a thickness in the ionized region
fraction_circular = -rho_average_circular / scc.e / n0

fraction_warpx_linear = 0.41014984 # result from WarpX simulation
fraction_warpx_circular = 0.502250841 # result from WarpX simulation

error_fraction_linear = np.abs( ( fraction_linear - fraction_warpx_linear ) / fraction_warpx_linear )
error_fraction_circular = np.abs( ( fraction_circular - fraction_warpx_circular ) / fraction_warpx_circular )

tolerance_higher = 0.15
tolerance_lower = 0.001
print(f"fraction_warpx_linear = {fraction_warpx_linear}")
print(f"fraction_hipace_linear = {fraction_linear}")
print(f"fraction_warpx_circular = {fraction_warpx_circular}")
print(f"fraction_hipace_circular = {fraction_circular}")

# in-situ diagnostics for calculation of the temperature in all directions in circular polarization
insitu_path_linear = f'./{args.third}/reduced_elec.0000.txt'
ir_linear = diag.InSituReader(insitu_path_linear)
Tx2 = ir_linear.slice_data('[ux^2]')[0,0]*scc.m_e*scc.c**2/scc.e
Ty2 = ir_linear.slice_data('[uy^2]')[0,0]*scc.m_e*scc.c**2/scc.e
Tz2 = ir_linear.slice_data('[uz^2]')[0,0]*scc.m_e*scc.c**2/scc.e
temp_eV_linear = 1./3*(Tx2+Ty2+Tz2)

insitu_path_circular = f'./{args.fourth}/reduced_elec.0000.txt'
ir_circular = diag.InSituReader(insitu_path_circular)
Tx2 = ir_circular.slice_data('[ux^2]')[0,0]*scc.m_e*scc.c**2/scc.e
Ty2 = ir_circular.slice_data('[uy^2]')[0,0]*scc.m_e*scc.c**2/scc.e
Tz2 = ir_circular.slice_data('[uz^2]')[0,0]*scc.m_e*scc.c**2/scc.e
temp_eV_circular = 1./3*(Tx2+Ty2+Tz2)

# calculation of temperature with OpenPMD-viewer diagnostics
wf, _ = ts_linear.get_field(field='w_elec', iteration=iteration)
sum_wf = np.sum(wf, axis=(1,2))[0:izmax]
ux2_linear, _ = ts_linear.get_field(field='ux^2_elec', iteration=iteration)
uy2_linear, _ = ts_linear.get_field(field='uy^2_elec', iteration=iteration)
uz2_linear, _ = ts_linear.get_field(field='uz^2_elec', iteration=iteration)
ux2_average_linear = np.mean(np.sum(ux2_linear * wf, axis=(1, 2))[0:izmax] / sum_wf)
uy2_average_linear = np.mean(np.sum(uy2_linear * wf, axis=(1, 2))[0:izmax] / sum_wf)
uz2_average_linear = np.mean(np.sum(uz2_linear * wf, axis=(1, 2))[0:izmax] / sum_wf)
temp_diags_linear = 1./3*(ux2_average_linear + uy2_average_linear + uz2_average_linear)*scc.m_e*scc.c**2/scc.e

wf, _ = ts_circular.get_field(field='w_elec', iteration=iteration)
sum_wf = np.sum(wf, axis=(1,2))[0:izmax]
ux2_circular, _ = ts_circular.get_field(field='ux^2_elec', iteration=iteration)
uy2_circular, _ = ts_circular.get_field(field='uy^2_elec', iteration=iteration)
uz2_circular, _ = ts_circular.get_field(field='uz^2_elec', iteration=iteration)
ux2_average_circular = np.mean(np.sum(ux2_circular * wf, axis=(1, 2))[0:izmax] / sum_wf)
uy2_average_circular = np.mean(np.sum(uy2_circular * wf, axis=(1, 2))[0:izmax] / sum_wf)
uz2_average_circular = np.mean(np.sum(uz2_circular * wf, axis=(1, 2))[0:izmax] / sum_wf)
temp_diags_circular = 1./3*(ux2_average_circular + uy2_average_circular + uz2_average_circular)*scc.m_e*scc.c**2/scc.e

temp_eV_warpx_linear = 1.00286009
temp_eV_warpx_circular = 9.68224535

print(f"temperature (eV) WarpX, linear = {temp_eV_warpx_linear}")
print(f"temperature (eV) HiPACE++, linear with insitu diagnostics = {temp_eV_linear}")
print(f"temperature (eV) HiPACE++, linear with Open-PMD diagnostics = {temp_diags_linear}")

print(f"temperature (eV) WarpX, circular = {temp_eV_warpx_circular}")
print(f"temperature (eV) HiPACE++, circular with insitu diagnostics = {temp_eV_circular}")
print(f"temperature (eV) HiPACE++, circular with Open-PMD diagnostics = {temp_diags_circular}")

# Error of temperature calculation between insitu-diagnostics and Open-PMD diagnostics in HiPACE++
error_h_h_linear = np.abs( ( temp_eV_linear - temp_diags_linear ) / temp_eV_linear )
error_h_h_circular = np.abs( ( temp_eV_circular - temp_diags_circular ) / temp_eV_circular )

# Error of temperature calculation between HiPACE++ and WarpX
error_h_w_linear = np.abs( ( temp_eV_linear - temp_eV_warpx_linear ) / temp_eV_warpx_linear )
error_h_w_circular = np.abs( ( temp_eV_circular - temp_eV_warpx_circular ) / temp_eV_warpx_circular )

assert ( (error_fraction_linear < tolerance_higher) and \
         (error_fraction_circular < tolerance_higher) and \
         (error_h_h_linear < tolerance_lower) and \
         (error_h_h_linear < tolerance_lower) and \
         (error_h_w_linear < tolerance_higher) and \
         (error_h_w_circular < tolerance_higher)), \
         'Test laser_ionization did not pass'
