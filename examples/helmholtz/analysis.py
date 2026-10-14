#! /usr/bin/env python3

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Tianyun Long
# License: BSD-3-Clause-LBNL

import sys
sys.path.append('/Users/maxencethevenet/src/hipace/tools')

import argparse
import numpy as np
from scipy.constants import c, epsilon_0, e, m_e, pi, eV
import read_insitu_diagnostics as isd

parser = argparse.ArgumentParser(description='Compare laser propagation in vacuum with theory')
parser.add_argument('--mode',
                    dest='mode',
                    help='Mode for the Helmholtz solver. "averaged" or "unaveraged"')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

gamma = 587
emit = 0.1e-6
lu = 0.05
K = 0.5
nperiod = 40
ku = 2*np.pi/lu
lr = lu/2/gamma**2*(1+K**2/2)
omegar = 2*pi/lr * c
a_to_E = m_e*c*omegar/e

insitu_beam = isd.InSituReader( args.output_dir + '/insitu/reduced_beam.*.txt')
insitu_helm = isd.InSituReader( args.output_dir + '/insitu/reduced_helmholtz.*.txt')

emitx = insitu_beam.avg_data("emittance_x") / insitu_beam.avg_data("[ga]")
emity = insitu_beam.avg_data("emittance_y") / insitu_beam.avg_data("[ga]")
betax = insitu_beam.avg_data("[x^2]") / emitx
betay = insitu_beam.avg_data("[y^2]") / emity
alphax = -insitu_beam.avg_data('[x*ux]') / gamma / emitx
alphay = -insitu_beam.avg_data('[y*uy]') / gamma / emity

print("betax[0], betax[10], betax[-1]", betax[0], betax[10], betax[-1])
print("betay[0], betay[10], betay[-1]", betay[0], betay[10], betay[-1])

if args.mode == 'averaged':
    # Check beam optics
    assert np.isclose(betax[-1], betay[0], rtol=0.01)
    assert np.isclose(betay[-1], betax[0], rtol=0.01)

    # Check radiated energy
    Er_t = epsilon_0*insitu_helm.avg_data("[|a|^2]") * a_to_E**2/2
    energy_genesis_uJ = 456 # after 3 undulators
    print("x-ray energy = ", Er_t[-1]*1e6, 'uJ, benchmark = ', energy_genesis_uJ)
    assert np.isclose(Er_t[-1]*1e6, energy_genesis_uJ, rtol=0.1 )
else:
    # Note that unaveraged runs shorter
    # Check beam optics
    assert np.isclose(betax[-1], 9.2, rtol=0.01)
    assert np.isclose(betay[-1], 2.3, rtol=0.1)

    # Check radiated energy
    Er_t = epsilon_0*insitu_helm.avg_data("[|a|^2]")
    energy_genesis_uJ = 6 # after 2 undulators
    print("x-ray energy = ", Er_t[-1]*1e6, 'uJ, benchmark = ', energy_genesis_uJ)
    assert np.isclose(Er_t[-1]*1e6, energy_genesis_uJ, rtol=0.4 )
