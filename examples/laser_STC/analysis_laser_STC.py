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
from lasy.utils.laser_utils import get_dispersion, get_zeta, get_beta
from lasy.laser import Laser
from lasy.profiles import FromOpenPMDProfile

lambda0 = .8e-6          # Laser wavelength
k0 = 2 * scc.pi / lambda0

parser = argparse.ArgumentParser(description='Compare laser propagation in vacuum with theory')
parser.add_argument('--output-dir',
                    dest='output_dir',
                    default='diags/hdf5',
                    help='Path to the directory containing output files')
args = parser.parse_args()

profile = FromOpenPMDProfile(
    file_name=args.output_dir+ '/openpmd_000000.h5',
    envelope_name='laserEnvelope',
)

laser = Laser(
        dim="xyt",
        lo=(-10e-6, -10e-6, -6e-13),
        hi=(10e-6, 10e-6, +6e-13),
        npoints=(255, 255, 1000),
        profile=profile,
     )
_, phi2 = get_dispersion(laser.grid,laser.dim, omega0=2*scc.pi*scc.c/lambda0, order=2)
[ _, beta_y] = get_beta(laser.dim, laser.grid, k0)
[ _, zeta_y], [_, _] = get_zeta(laser.dim, laser.grid, k0)
print("phi2 theory:", 2.4e-27, "measured:", phi2)
print("zeta_y theory:", 2.4e-22, "measured:", zeta_y)
print("beta_y theory:", 3e-18, "measured:", beta_y)
assert (np.abs((phi2-2.4e-27)/2.4e-27)<0.1), 'Test phi2 did not pass'
assert (np.abs((zeta_y-2.4e-22)/2.4e-22)<0.1), 'Test zeta did not pass'
assert (np.abs((beta_y-3e-18)/3e-18)<0.2), 'Test beta did not pass'
