#! /usr/bin/env python3

# Copyright 2025
#
# This file is part of HiPACE++.
#
# Authors: Xingjian Hui
# License: BSD-3-Clause-LBNL

import argparse
import numpy as np
import scipy.constants as scc
from lasy.utils.laser_utils import get_phi2, get_zeta, get_beta
from lasy.laser import Laser
from lasy.profiles import FromOpenPMDProfile

lambda0 = .6e-6          # Laser wavelength
k0 = 2 * scc.pi / lambda0

parser = argparse.ArgumentParser(description='Compare laser propagation in vacuum with theory')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

profile = FromOpenPMDProfile(
    file_name=args.output_dir+ '/openpmd_000000.h5',
    envelope_name = 'laserEnvelope',
)

laser = Laser(
        dim="xyt",
        lo=(-10e-6, -10e-6, -0e-15),
        hi=(10e-6, 10e-6, +200e-15),
        npoints=(255, 255, 730),
        profile=profile,
     )

phi2 = get_gdd(laser.dim, laser.grid, omega0 = 2 * scc.pi*scc.c/lambda0)
[beta_x, beta_y] = get_beta(laser.dim, laser.grid, k0)
[zeta_x, zeta_y], [nu_x, nu_y] = get_zeta(laser.dim, laser.grid, k0)
print("phi2 theory:", 3e-28, "measured:", phi2)
print("zeta_y theory:", 2.4e-22, "measured:", zeta_y)
print("beta_y theory:", 3e-18, "measured:", beta_y)
assert (np.abs((phi2-3e-28)/2.4e-24)<0.1), 'Test phi2 did not pass'
assert (np.abs((zeta_y-2.4e-22)/2.4e-22)<0.1), 'Test zeta did not pass'
assert (np.abs((beta_y-3e-18)/3e-18)<0.2), 'Test beta did not pass'
