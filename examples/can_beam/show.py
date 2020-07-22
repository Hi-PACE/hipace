#! /usr/bin/env python

# Plots Ex field and particles at the end of the run,
# and save image to img.png
# usage: ./show.py

import yt ; yt.funcs.mylog.setLevel(50)
import matplotlib.pyplot as plt
from yt.frontends.boxlib.data_structures import AMReXDataset

ds = AMReXDataset('plt00001')

plot_yt = True
plot_plt = True

if plot_yt:
    # Plot 1 field with yt.sliceplot
    sl = yt.SlicePlot(ds, 2, 'jz', aspect=1)
    sl.annotate_particles(width=(1., 'm'), p_size=2, ptype='plasma', col='black')
    sl.annotate_particles(width=(1., 'm'), p_size=2, ptype='beam', col='red')
    sl.annotate_grids()
    sl.save('./yt_img.png')

if plot_plt:
    # Plot all fields and particles in 1 figure
    nx = 3
    ny = 3
    # Get field quantities
    all_data_level_0 = ds.covering_grid(level=0,left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    Dx = ds.domain_width/ds.domain_dimensions
    extent = [ds.domain_left_edge[ds.dimensionality-1], ds.domain_right_edge[ds.dimensionality-1],
              ds.domain_left_edge[0], ds.domain_right_edge[0] ]
    # Get particle quantities
    ad = ds.all_data()
    x = ad['beam', 'particle_position_x'].v
    z = ad['beam', 'particle_position_z'].v
    # Loop over fields and plot them
    plt.figure(figsize=(10, 10))
    grid_fields = [i for i in ds.field_list if i[0] == 'boxlib']
    for count, field in enumerate(grid_fields):
        plt.subplot(nx,ny,count+1)
        plt.title(field[1])
        F = all_data_level_0[field].v.squeeze()[:,ds.domain_dimensions[1]//2,:]
        plt.imshow(F, extent=extent, aspect='auto')
        plt.colorbar()
        plt.xlabel('z')
        plt.ylabel('x')
    plt.subplot(nx,ny,1)
    plt.scatter(z,x,s=.002,c='k')
    plt.tight_layout()
    plt.savefig('./plt_img.png', bbox_inches='tight')
