import math
import openpmd_api as io
import numpy as np
from numpy import random

n = 1000000
beam_density = 3
beam_position_mean = [0, 0, 0]
beam_position_std = [0.3, 0.3, 1.4]
beam_u_mean = [0, 0, 2000]
beam_u_std = [0, 0, 0]

single_charge = (beam_density * beam_position_std[0] * beam_position_std[1] *
                 beam_position_std[2] * np.sqrt(2. * math.pi)**3 / n)

data = np.zeros([6,n],dtype=np.float64)

for i in [0,1,2]:
    data[i]=random.normal(beam_position_mean[i],beam_position_std[i],n)
    data[i+3]=random.normal(beam_u_mean[i],beam_u_std[i],n)

series = io.Series("beam_%05T.h5", io.Access.create)

i = series.iterations[0]

partikel = i.particles["Electrons"]

dataset = io.Dataset(data[0].dtype,data[0].shape)

partikel["r"].unit_dimension = {
    io.Unit_Dimension.L:  1,
}

partikel["u"].unit_dimension = {
    io.Unit_Dimension.L:  1,
    io.Unit_Dimension.T: -1,
}

partikel["q"].unit_dimension = {
    io.Unit_Dimension.I:  1,
    io.Unit_Dimension.T:  1,
}

partikel["m"].unit_dimension = {
    io.Unit_Dimension.M:  1,
}

for j,k,m in [["r","x",0],["r","y",1],["r","z",2]]:
    partikel[j][k].reset_dataset(dataset)
    partikel[j][k].store_chunk(data[m])
    partikel[j][k].unit_SI = 1.e-5

for j,k,m in [["u","x",3],["u","y",4],["u","z",5]]:
    partikel[j][k].reset_dataset(dataset)
    partikel[j][k].store_chunk(data[m])
    partikel[j][k].unit_SI = 1

partikel["q"]["q"].reset_dataset(dataset)
partikel["q"]["q"].make_constant(single_charge)
partikel["q"]["q"].unit_SI = 1.602176634e-19 * 2.8239587008591567e23 * (1.e-5)**3

partikel["m"]["m"].reset_dataset(dataset)
partikel["m"]["m"].make_constant(single_charge)
partikel["m"]["m"].unit_SI = 1.602176634e-19 * 2.8239587008591567e23 * (1.e-5)**3 / 1.7588e11

series.flush()

del series
