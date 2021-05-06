#! /usr/bin/env python3

# This Python analysis script is part of the code Hipace
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
import openpmd_api as io

parser = argparse.ArgumentParser(description='Script to analyze the correctness of beam from_file')
parser.add_argument('--beam-py',
                    dest='beam_py',
                    required=True,
                    help='Path to the data of python beam')
parser.add_argument('--beam-out1',
                    dest='beam_out1',
                    required=True,
                    help='Path to the data of the first run')
parser.add_argument('--beam-out2',
                    dest='beam_out2',
                    required=True,
                    help='Path to the data of the restart run')
parser.add_argument('--SI',
                    dest='in_SI_units',
                    action='store_true',
                    default=False,
                    help='SI or normalized units')
args = parser.parse_args()

ser_py = io.Series(args.beam_py,io.Access.read_only)
ser_o1 = io.Series(args.beam_out1,io.Access.read_only)
ser_o2 = io.Series(args.beam_out2,io.Access.read_only)

par_py = ser_py.iterations[0].particles["Electrons"]
par_o1 = ser_o1.iterations[0].particles["beam"]
par_o2 = ser_o2.iterations[0].particles["beam"]

all_comps = [["r", "position", [["x", "x"], ["y", "y"], ["z", "z"]]], \
             ["u", "momentum", [["x", "x"], ["y", "y"], ["z", "z"]]], \
             ["q", "weighting", [["q", io.Mesh_Record_Component.SCALAR]]]]

units_py = {"r" : 1. , "u": scc.c * scc.m_e, "q": 1./scc.e}

for comp in all_comps:
    for axes in comp[2]:
        data_py = par_py[comp[0]][axes[0]]
        data_o1 = par_o1[comp[1]][axes[1]]
        data_o2 = par_o2[comp[1]][axes[1]]
        arr_py = data_py.load_chunk()
        arr_o1 = data_o1.load_chunk()
        arr_o2 = data_o2.load_chunk()
        ser_py.flush()
        ser_o1.flush()
        ser_o2.flush()

        if args.in_SI_units:
            arr_py *= data_py.unit_SI * units_py[comp[0]]
            arr_o1 *= data_o1.unit_SI
            arr_o2 *= data_o2.unit_SI
        elif comp[0] == "q":
            arr_py *= data_py.unit_SI * units_py[comp[0]] \
                    * np.sqrt(par_py.get_attribute("Hipace++_Plasma_Density"))
            arr_o1 *= data_o1.get_attribute("Hipace++_reference_unitSI") \
                    * np.sqrt(par_o1.get_attribute("Hipace++_Plasma_Density"))
            arr_o2 *= data_o2.get_attribute("Hipace++_reference_unitSI") \
                    * np.sqrt(par_o2.get_attribute("Hipace++_Plasma_Density"))

        are_equal_py_o1 = np.all(np.isclose(np.sort(arr_py), np.sort(arr_o1), rtol=1.e-8))
        if not are_equal_py_o1:
            print("Python generated beam file and Hipace output are not equal for", \
                  comp[1], "component")
        are_equal_o1_o2 = np.all(np.isclose(np.sort(arr_o1), np.sort(arr_o2), rtol=1.e-8))
        if not are_equal_o1_o2:
            print("Hipace initial beam output and output after restarting are not equal for", \
                  comp[1], "component")

        assert(are_equal_py_o1)
        assert(are_equal_o1_o2)

print("All beam files are equal between the initial python generated file,", \
      "the first Hipace output, and the Hipace output after restarting.")

del ser_py
del ser_o1
del ser_o2
