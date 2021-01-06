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

single_charge = (beam_density * beam_position_std[0] * beam_position_std[1] *
                 beam_position_std[2] * np.sqrt(2. * math.pi)**3 / n)

data = np.zeros([6,n],dtype=np.float64)

for i in [0,1,2]:
    data[i]=random.normal(beam_position_mean[i],beam_position_std[i],n)
    data[i+3]=random.normal(beam_u_mean[i],beam_u_std[i],n)

series = io.Series("beam_%05T.h5", io.Access.create)

i = series.iterations[0]

particel = i.particles["Electrons"]

dataset = io.Dataset(data[0].dtype,data[0].shape)

particel["r"].unit_dimension = {
    io.Unit_Dimension.L:  1,
}

particel["u"].unit_dimension = {
    io.Unit_Dimension.L:  1,
    io.Unit_Dimension.T: -1,
}

particel["q"].unit_dimension = {
    io.Unit_Dimension.I:  1,
    io.Unit_Dimension.T:  1,
}

particel["m"].unit_dimension = {
    io.Unit_Dimension.M:  1,
}

for k,m in [["x",0],["y",1],["z",2]]:
    particel["r"][k].reset_dataset(dataset)
    particel["r"][k].store_chunk(data[m])
    particel["r"][k].unit_SI = kp_inv

for k,m in [["x",3],["y",4],["z",5]]:
    particel["u"][k].reset_dataset(dataset)
    particel["u"][k].store_chunk(data[m])
    particel["u"][k].unit_SI = 1

particel["q"]["q"].reset_dataset(dataset)
particel["q"]["q"].make_constant(single_charge)
particel["q"]["q"].unit_SI = constants.e * plasma_density * kp_inv**3

particel["m"]["m"].reset_dataset(dataset)
particel["m"]["m"].make_constant(single_charge)
particel["m"]["m"].unit_SI = constants.m_e * plasma_density * kp_inv**3

series.flush()

del series
