#! /usr/bin/env python3

# Copyright 2026
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries


ts1 = OpenPMDTimeSeries('initial_sim')
ts2 = OpenPMDTimeSeries('restart_sim')

exmby1 = ts1.get_field(field="ExmBy", iteration=0)[0]
exmby2 = ts2.get_field(field="ExmBy", iteration=0)[0]

error = np.max(np.abs(exmby1[:50] - exmby2[50:])) / np.max(np.abs(exmby1[:50]))
print("error =", error)
assert(error < 1.e-8)

del ts1
del ts2
