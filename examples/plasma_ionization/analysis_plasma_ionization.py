import argparse
import numpy as np
import math
from openpmd_viewer import OpenPMDTimeSeries
import scipy.constants as scc

parser = argparse.ArgumentParser(
    description='Script to analyze the equality of two simulations')
parser.add_argument('--diags_grid',
                    dest='first',
                    required=True,
                    help='Path to the directory containing output files')
parser.add_argument('--diags_particle',
                    dest='second',
                    required=True,
                    help='Path to the directory containing output files')
args = parser.parse_args()
ts1 = OpenPMDTimeSeries(args.first)
ts2 = OpenPMDTimeSeries(args.second)
Ar2, m2 = ts2.get_field(field='n_ion_ionlev_1', iteration=0)
Ar1, m1 = ts1.get_field(field='grid_ionization_w_ion_1', iteration=0)
ep = np.abs(np.sum(Ar2)*m2.dz*m2.dx*m2.dy-np.sum(Ar1)*m1.dx*m1.dy*m1.dz)/(np.sum(Ar1)*m1.dx*m1.dy*m1.dz)
tolerance = 0.05
assert ep < tolerance, 'Test grid_ionization did not pass'