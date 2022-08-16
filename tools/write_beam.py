# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, Severin Diederichs
# License: BSD-3-Clause-LBNL

import math
import openpmd_api as io
import numpy as np
from numpy import random
from scipy import constants

n = 1000000
beam_density = 3.
plasma_density = 2.8239587008591567e23
beam_position_mean = [0, 0, 0]
beam_position_std = [0.3, 0.3, 1.41]
beam_u_mean = [0, 0, 2000]
beam_u_std = [0, 0, 0]

kp_inv = constants.c / constants.e * math.sqrt(constants.epsilon_0 * constants.m_e / plasma_density)

single_weight = (beam_density * beam_position_std[0] * beam_position_std[1] *
                 beam_position_std[2] * np.sqrt(2. * math.pi)**3 / n)

rng = random.default_rng(seed=0)
data = np.zeros([6,n],dtype=np.float64)

for i in [0,1,2]:
    data[i]=rng.normal(beam_position_mean[i],beam_position_std[i],n)
    data[i+3]=rng.normal(beam_u_mean[i],beam_u_std[i],n)

series = io.Series("beam_%05T.h5", io.Access.create)

i = series.iterations[0]

particle = i.particles["Electrons"]

particle.set_attribute("HiPACE++_Plasma_Density", plasma_density)

dataset = io.Dataset(data[0].dtype,data[0].shape)

particle["position"].unit_dimension = {
    io.Unit_Dimension.L:  1,
}

particle["momentum"].unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.L:  1,
    io.Unit_Dimension.T: -1,
}

particle["charge"].unit_dimension = {
    io.Unit_Dimension.I:  1,
    io.Unit_Dimension.T:  1,
}

particle["mass"].unit_dimension = {
    io.Unit_Dimension.M:  1,
}

for k,m in [["x",0],["y",1],["z",2]]:
    particle["position"][k].reset_dataset(dataset)
    particle["position"][k].store_chunk(data[m])
    particle["position"][k].unit_SI = kp_inv

for k,m in [["x",3],["y",4],["z",5]]:
    particle["momentum"][k].reset_dataset(dataset)
    particle["momentum"][k].store_chunk(data[m])
    particle["momentum"][k].unit_SI = constants.m_e * constants.c

particle["charge"]["charge"].reset_dataset(dataset)
particle["charge"]["charge"].make_constant(single_weight)
particle["charge"]["charge"].unit_SI = constants.e * plasma_density * kp_inv**3

particle["mass"]["mass"].reset_dataset(dataset)
particle["mass"]["mass"].make_constant(single_weight)
particle["mass"]["mass"].unit_SI = constants.m_e * plasma_density * kp_inv**3

series.flush()

del series
