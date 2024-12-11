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
from lasy.laser import Laser

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
profile = FromOpenPMDProfile(path=args.output_dir,iteration=0,pol=[1,0],field='laserEnvelope', is_envelope=True, prefix='openpmd')
laser = Laser(
        dim="xyt",
        lo=(np.min(profile.axes['x']), np.min(profile.axes['y']), np.min(profile.axes['t'])),
        hi=(np.max(profile.axes['x']), np.max(profile.axes['y']), np.max(profile.axes['t'])),
        npoints=(255, 255, 200),
        profile=profile,
     )

k0 = 2 * scc.pi / 0.6e-6
stc=get_STC(laser.dim,laser.grid,k0)
print('zeta is ')
print(stc['zeta_x'])
print('beta is ')
print(stc['beta_x'])
assert(np.abs(stc['phi2'] - 2.4e-19) / 2.4e-19 < 1e-2)
assert(np.abs(stc['beta_x'] - 3e-18) / 3e-18 < 1e-2)
assert(np.abs(stc['zeta_x'] - 2.4e-24) / 2.4e-24 < 1e-2)
