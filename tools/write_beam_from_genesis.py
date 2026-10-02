import openpmd_api as io
import numpy as np
from scipy.constants import c, m_e, e
import h5py
import os.path
import sys

filename_in = 'beam.genesis_8192.h5'
filename_out = 'beam.openpmd_8192'

if os.path.isfile(filename_out + '_00000.h5'):
    print("file exists, don't do anything")
    sys.exit()

print(filename_in, filename_out)

f = h5py.File(filename_in, 'r')
total_charge = 300e-12

dz = np.array(f['slicespacing'])
x = []  ; y = [] ;  z = []
px = [] ; py = [] ; pz = []
for i in range(np.array(f['slicecount'])[0]):
    slicename = 'slice' + str(i+1).zfill(6)
    if f[slicename]['current'][0] > 0:
        x = np.append(x, np.array(f[slicename]['x']), axis=0)
        y = np.append(y, np.array(f[slicename]['y']), axis=0)
        theta = np.array(f[slicename]['theta'])
        zs = -(i+theta/2/np.pi) * dz
        z = np.append(z, zs, axis=0)
        gammas = np.array(f[slicename]['gamma'])
        uxs = np.array(f[slicename]['px'])
        uys = np.array(f[slicename]['py'])
        uzs = np.sqrt( gammas**2 - 1 - uxs**2 - uys**2 )
        px = np.append(px, uxs, axis=0)
        py = np.append(py, uys, axis=0)
        pz = np.append(pz, uzs, axis=0)

n = px.size
single_weight = total_charge / n / e
print(single_weight)

data = np.zeros([6,n],dtype=np.float64)
data[0] = x
data[1] = y
data[2] = z
data[3] = px
data[4] = py
data[5] = pz

series = io.Series(filename_out + "_%05T.h5", io.Access.create)

i = series.iterations[0]

particle = i.particles["Electrons"]

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
    particle["position"][k].unit_SI = 1

for k,m in [["x",3],["y",4],["z",5]]:
    particle["momentum"][k].reset_dataset(dataset)
    particle["momentum"][k].store_chunk(data[m])
    particle["momentum"][k].unit_SI = m_e * c

SCALAR = io.Mesh_Record_Component.SCALAR

particle["charge"][SCALAR].reset_dataset(dataset)
particle["charge"][SCALAR].make_constant(single_weight)
particle["charge"][SCALAR].unit_SI = e

particle["mass"][SCALAR].reset_dataset(dataset)
particle["mass"][SCALAR].make_constant(single_weight)
particle["mass"][SCALAR].unit_SI = m_e

series.flush()

del series
