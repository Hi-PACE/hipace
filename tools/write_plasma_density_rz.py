#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Create an openPMD file containing the plasmas density with which
the HiPACE++ plasma particles should be initialized
"""

import numpy as np
import openpmd_api as io

# Define density as a function of r, z, modes using numpy syntax
# parabolic channel in r, with a ramp and plateau in z
on_axis_density = 1e24  # m^-3
channel_radius = 40e-6  # m
ramp_length = 60e-6  # m
nmodes = 1
# - Define the grid
r_1d = np.linspace(-10e-6, 100e-6, 200)
z_1d = np.linspace(0, 500e-6, 200)
m_1d = np.array(list(range(2*nmodes-1)))

m, z, r = np.meshgrid(m_1d, z_1d, r_1d, indexing="ij")

# - Define density as a function of x, y, z
density_data = (
    on_axis_density
    * (1 + r**2 / channel_radius**2)
    * np.where(z < ramp_length, z / ramp_length, 1)
    * np.where((m == 0), 1, 0)
)

# create openpmd file
series = io.Series("example-density-rz.h5", io.Access.create)
# only 1 iteration needed
it = series.iterations[0]
# set meta information
density = it.meshes["density"]
density.grid_spacing = np.array(
    [(z_1d[-1] - z_1d[0]) / (len(z_1d) - 1),
     (r_1d[-1] - r_1d[0]) / (len(r_1d) - 1)]
)
density.grid_global_offset = [z_1d[0], r_1d[0]]
density.axis_labels = ["z", "r"]
density.geometry = io.Geometry.thetaMode
density.unit_dimension = {
    io.Unit_Dimension.L: -3,
}

# label
density_d = density[io.Mesh_Record_Component.SCALAR]
density_d.position = [0, 0, 0]

dataset = io.Dataset(density_data.dtype, density_data.shape)
density_d.reset_dataset(dataset)
density_d.store_chunk(density_data)
series.flush()

del series
