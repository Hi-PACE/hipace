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
from lasy.utils.laser_utils import get_Beta, get_Phi2, get_Zeta
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
        npoints=(511, 255, 200),
        profile=profile,
     )

k0 = 2 * scc.pi / 0.6e-6
Phi2, phi2 = get_Phi2(laser.dim, laser.grid)
[zeta_x, zeta_y]  = get_Zeta(laser.dim, laser.grid, k0)
[beta_x, beta_y] = get_Beta(laser.dim, laser.grid, k0)

print('phi2 is ')
print(phi2)
print('zeta is ')
print([zeta_x, zeta_y])
print('beta is ')
print([beta_x, beta_y])
#np.testing.assert_approx_equal(phi2, 2.4e-24, significant=2)
np.testing.assert_approx_equal(zeta_y, 2.4e-22, significant=2)
np.testing.assert_approx_equal(beta_y, 3e-18, significant=2)
