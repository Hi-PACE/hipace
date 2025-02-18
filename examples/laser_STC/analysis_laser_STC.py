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

profile = FromOpenPMDProfile (path = args.output_dir, iteration = 0, pol=[1,0], field='laserEnvelope'\
                              ,coord='', is_envelope=True, prefix='openpmd')
laser = Laser(
        dim="xyt",
        lo=(-15e-6, -15e-6, -30e-15),
        hi=(15e-6, 15e-6, +30e-15),
        npoints=(127, 127, 200),
        profile=profile,
     )

Phi2,phi2 = get_phi2(laser.dim, laser.grid)
[beta_x, beta_y] = get_beta(laser.dim, laser.grid, k0)
[zeta_x, zeta_y], [nu_x, nu_y] = get_zeta(laser.dim, laser.grid, k0)

assert ((phi2-2.4e-24)/2.4e-24<0.01), 'Test phi2 did not pass'
assert ((zeta_y-2.4e-22)/2.4e-22<0.01), 'Test zeta did not pass'
assert ((beta_y-3e-18)/3e-18<0.01), 'Test zeta did not pass'
