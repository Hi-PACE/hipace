#! /usr/bin/env python3

# Copyright 2026
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries


ts = OpenPMDTimeSeries('histogram.2Rank')

hist_uz1 = ts.get_field(field="beam_hist_z_uz1", iteration=ts.iterations[-1])[0]
hist_uz2 = ts.get_field(field="beam_hist_z_uz2", iteration=ts.iterations[-1])[0]

error = np.max(np.abs(hist_uz1 - hist_uz2.T)) / np.max(np.abs(hist_uz1))
print("error =", error)
assert(error < 1.e-20)

del ts

