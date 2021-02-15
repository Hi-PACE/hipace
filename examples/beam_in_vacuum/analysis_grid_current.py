#! /usr/bin/env python3

# This Python analysis script is part of the code Hipace
#
# It calculates the sum of jz. The beam current and the grid current should cancel each other.

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

ts = OpenPMDTimeSeries('./diags/h5/')

# Load Hipace data for jz
jz_sim, jz_info = ts.get_field(field='jz', iteration=1)

# Assert that the grid current and the beam current cancel each other
error_jz = np.sum( (jz_sim)**2)
print("sum of jz**2: " + str(error_jz) + " (tolerance = 3e-3)")

assert(error_jz < 3e-3)
