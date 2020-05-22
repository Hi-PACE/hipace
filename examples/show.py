#! /usr/bin/env python

# Plots Ex field and particles at the end of the run,
# and save image to img.png
# usage: ./show.py

import yt ; yt.funcs.mylog.setLevel(50)
import matplotlib.pyplot as plt
from yt.frontends.boxlib.data_structures import AMReXDataset

ds = AMReXDataset('plt00001')

sl = yt.SlicePlot(ds, 2, 'Ex', aspect=1)
sl.annotate_particles(width=(1., 'm'), p_size=2, ptype='plasma', col='black')
sl.annotate_particles(width=(1., 'm'), p_size=2, ptype='beam', col='red')
sl.save('./img.png')
