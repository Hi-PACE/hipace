#! /usr/bin/env python3

# Copyright 2024
#
# This file is part of HiPACE++.
#
# Authors: Xingjian Hui
# License: BSD-3-Clause-LBNL

import argparse
import numpy as np
import scipy.constants as scc
from openpmd_viewer.addons import LpaDiagnostics
from lasy.utils.laser_utils import get_STC
from lasy.profiles import FromOpenPMDProfile

parser = argparse.ArgumentParser(description = 'Verify the chirp initialization')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
parser.add_argument('--chirp_type',
                    dest='chirp_type',
                    default='phi2',
                    help='Type of the initialized chirp')
args = parser.parse_args()
print(args.output_dir)
profile = FromOpenPMDProfile(path=' ',iteration=0,pol=[1,0],field='laserEnvelope', is_envelope=True, prefix=args.output_dir)
laser = Laser(
        dim="xyt",
        lo=(-15e-6, -15e-6, -30e-15),
        hi=(15e-6, 15e-6, +30e-15),
        npoints=(255, 255, 50),
        profile=profile,
     )

k0 = 2 * scc.pi / lambda0
stc=get_STC(laser.grid,laser.dim,k0)
print('zeta is ')
print(stc['zeta_x'])
print('beta is ')
print(stc['beta_x'])
assert(np.abs(stc['phi2'] - 2.4e-19) / 2.4e-19 < 1e-2)
assert(np.abs(stc['beta_x'] - 3e-18) / 3e-18 < 1e-2)
assert(np.abs(stc['zeta_x'] - 2.4e-24) / 2.4e-24 < 1e-2)
