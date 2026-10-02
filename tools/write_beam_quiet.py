import openpmd_api as io
import numpy as np
import scipy.special
import scipy.constants as sc

gamma = 299.96 / 0.511
Lbeam = 18e-6
w0 = 22.6e-6
emit = 0.1e-6
lu = 0.05
K = 0.5
lr = lu/2/gamma**2*(1+K**2/2)
part_per_bin = 256
bins_per_lr = 4
total_charge = 300e-12

num_slices = int(round(Lbeam/lr))
n = part_per_bin * bins_per_lr * num_slices
single_weight = int(total_charge / n / sc.e)
dzs = lr / bins_per_lr
data = np.zeros([7,n],dtype=np.float64)

def hammersley_uniform(n, seed, base):
    seq = np.zeros(n)
    for i in range(n):
        res, denom = 0.0, 1.0
        ii = i + 1 + abs(seed)
        while ii > 0:
            denom *= base
            res += (ii % base) / denom
            ii //= base
        seq[i] = res
    return seq

def hammersley_normal(n, seed, base):
    return np.sqrt(2) * scipy.special.erfinv(hammersley_uniform(n, seed, base) * 2 -1)

def remove_noise_gauss(x):
    x = (x - np.mean(x)) / np.std(x)
    return x

def remove_noise_gauss_2d(x, y):
    c = (np.mean(x*y) - np.mean(x)*np.mean(y)) / np.std(x)**2
    y = y - np.mean(y) - c * x
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    return x, y

for zslice in range(num_slices):
    for zbin in range(bins_per_lr):
        idx = zbin + zslice * bins_per_lr * m
        oidx = zslice * bins_per_lr * m
        m = part_per_bin
        zs = -num_slices*lr + (zbin + zslice * bins_per_lr) * dzs
        begin = idx
        end = idx + m * bins_per_lr
        obegin = oidx
        oend = oidx + m * bins_per_lr

        if zbin == 0:
            seed = int(1e9 * np.random.uniform(0, 1, 1))
            data[0, begin:end:bins_per_lr] = hammersley_normal(m, seed, 5)
            data[1, begin:end:bins_per_lr] = hammersley_normal(m, seed, 7)
            data[2, begin:end:bins_per_lr] = hammersley_uniform(m, seed, 2)
            data[3, begin:end:bins_per_lr] = hammersley_normal(m, seed, 11)
            data[4, begin:end:bins_per_lr] = hammersley_normal(m, seed, 13)
            data[5, begin:end:bins_per_lr] = hammersley_normal(m, seed, 3)

            data[5, begin:end:bins_per_lr] = remove_noise_gauss(data[5, begin:end:bins_per_lr])
            data[0, begin:end:bins_per_lr], data[3, begin:end:bins_per_lr] = remove_noise_gauss_2d(data[0, begin:end:bins_per_lr], data[3, begin:end:bins_per_lr])
            data[1, begin:end:bins_per_lr], data[4, begin:end:bins_per_lr] = remove_noise_gauss_2d(data[1, begin:end:bins_per_lr], data[4, begin:end:bins_per_lr])

            data[0, begin:end:bins_per_lr] = w0 * data[0, begin:end:bins_per_lr]
            data[1, begin:end:bins_per_lr] = w0 * data[1, begin:end:bins_per_lr]
            data[2, begin:end:bins_per_lr] = zs + dzs * data[2, begin:end:bins_per_lr]
            data[3, begin:end:bins_per_lr] = emit/w0 * data[3, begin:end:bins_per_lr]
            data[4, begin:end:bins_per_lr] = emit/w0 * data[4, begin:end:bins_per_lr]
            data[5, begin:end:bins_per_lr] = gamma + 0 * data[5, begin:end:bins_per_lr]
        else:
            data[0, begin:end:bins_per_lr] = data[0, obegin:oend:bins_per_lr]
            data[1, begin:end:bins_per_lr] = data[1, obegin:oend:bins_per_lr]
            data[2, begin:end:bins_per_lr] = data[2, obegin:oend:bins_per_lr] + zbin * dzs
            data[3, begin:end:bins_per_lr] = data[3, obegin:oend:bins_per_lr]
            data[4, begin:end:bins_per_lr] = data[4, obegin:oend:bins_per_lr]
            data[5, begin:end:bins_per_lr] = data[5, obegin:oend:bins_per_lr]

# Calculate weight based on z position

# Flattop
data[6, :] = single_weight

# Gauss
# zmean = -100 * lr
# zstd = 30 * lr
# data[6, :] = single_weight * np.exp(-(data[2, :] - zmean)**2 / zstd**2)

# Add shot noise
for zslice in range(num_slices):
    m = part_per_bin
    d_start = zslice * bins_per_lr * m
    d_end = (zslice + 1) * bins_per_lr * m
    avg_single_wieght = np.mean(data[6, d_start:d_end])
    nbl = max(avg_single_wieght, 1) * bins_per_lr
    tmp = np.zeros(bins_per_lr*m)
    for ih in range((bins_per_lr-1)//2):
        phi = np.repeat(2 * np.pi * np.random.uniform(0, 1, m), bins_per_lr)
        an = np.repeat(np.fmod(np.sqrt( - np.log(np.random.uniform(0, 1, m))/nbl)*2/(ih+1), 2*np.pi), bins_per_lr)
        tmp -= an*np.sin( data[2, d_start:d_end] * (2* np.pi / lr) * (ih+1) + phi)
    data[2, d_start:d_end] += tmp * lr / (2 * np.pi)


series = io.Series("beam_quiet_%05T.h5", io.Access.create)

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
    particle["momentum"][k].unit_SI = sc.m_e * sc.c

SCALAR = io.Mesh_Record_Component.SCALAR

particle["weighting"][SCALAR].reset_dataset(dataset)
particle["weighting"][SCALAR].store_chunk(data[6])
particle["weighting"][SCALAR].unit_SI = 1

particle["charge"][SCALAR].reset_dataset(dataset)
particle["charge"][SCALAR].make_constant(single_weight)
particle["charge"][SCALAR].unit_SI = sc.e

particle["mass"][SCALAR].reset_dataset(dataset)
particle["mass"][SCALAR].make_constant(single_weight)
particle["mass"][SCALAR].unit_SI = sc.m_e

series.flush()

del series
