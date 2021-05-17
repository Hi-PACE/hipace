#! /usr/bin/env python3

# This Python analysis script is part of the code HiPACE++
#
# It is used in the beam from_file test and compares three beams,
# one from the write_beam script, two from hipace,
# with each other and asserts if they are different.

import matplotlib.pyplot as plt
import scipy.constants as scc
import matplotlib
import sys
import numpy as np
import math
import argparse
import openpmd_api as io # to have access to Attribute Metadata

parser = argparse.ArgumentParser(description='Script to analyze the correctness of beam from_file')
parser.add_argument('--beam-py',
                    dest='beam_py',
                    default='',
                    help='Path to the data of python beam')
parser.add_argument('--beam-out1',
                    dest='beam_out1',
                    default='',
                    help='Path to the data of the first run')
parser.add_argument('--beam-out2',
                    dest='beam_out2',
                    default='',
                    help='Path to the data of the restart run')
parser.add_argument('--SI',
                    dest='in_SI_units',
                    action='store_true',
                    default=False,
                    help='SI or normalized units')
args = parser.parse_args()

beam_ser = [None, None]
beam_par = [None, None]

if args.beam_py != '' and args.beam_out1 != '' and args.beam_out2 == '':
    beam_ser[0] = io.Series(args.beam_py,io.Access.read_only)
    beam_ser[1] = io.Series(args.beam_out1,io.Access.read_only)
    beam_par[0] = beam_ser[0].iterations[0].particles["Electrons"]
    beam_par[1] = beam_ser[1].iterations[0].particles["beam"]
    beam_type = [0, 1]

elif args.beam_py == '' and args.beam_out1 != '' and args.beam_out2 != '':
    beam_ser[0] = io.Series(args.beam_out1,io.Access.read_only)
    beam_ser[1] = io.Series(args.beam_out2,io.Access.read_only)
    beam_par[0] = beam_ser[0].iterations[0].particles["beam"]
    beam_par[1] = beam_ser[1].iterations[0].particles["beam"]
    beam_type = [1, 1]

else:
    raise AssertionError("Invalid input")

all_comps = [["r", "position", [["x", "x"], ["y", "y"], ["z", "z"]]], \
             ["u", "momentum", [["x", "x"], ["y", "y"], ["z", "z"]]], \
             ["q", "weighting", [["q", io.Mesh_Record_Component.SCALAR]]]]

units_py = {"r" : 1. , "u": scc.c * scc.m_e, "q": 1./scc.e}

for comp in all_comps:
    for axes in comp[2]:
        beam_arr = [None, None]
        for i in range(2):
            beam_data = beam_par[i][comp[beam_type[i]]][axes[beam_type[i]]]
            beam_arr[i] = beam_data.load_chunk()
            beam_ser[i].flush()

            if args.in_SI_units:
                beam_arr[i] *= beam_data.unit_SI
                if beam_type[i] == 0:
                    beam_arr[i] *= units_py[comp[0]]
            elif comp[0] == "q":
                beam_arr[i] *= np.sqrt(beam_par[i].get_attribute("HiPACE++_Plasma_Density"))
                if beam_type[i] == 0:
                    beam_arr[i] *= beam_data.unit_SI * units_py[comp[0]]
                else:
                    beam_arr[i] *= beam_data.get_attribute("HiPACE++_reference_unitSI")

        are_equal = np.all(np.isclose(np.sort(beam_arr[0]), np.sort(beam_arr[1]), rtol=1.e-8))
        assert are_equal, f"The two beams are not equal for {comp[1]} component"

print("The two beam files are equal.")

del beam_ser
