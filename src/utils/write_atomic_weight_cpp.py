#! /usr/bin/env python

"""
This python script reads atomic masses tables in relative_atomic_masses.txt
(generated from the NIST website) and extracts
standard atomic weights into C++ file AtomicWeightTable.H.
"""

import re, os
import numpy as np

# get data
filename = os.path.join( '.', 'relative_atomic_masses.txt' )
with open(filename) as f:
    text_data = f.read()

regex_command = 'Atomic Number = \d+\n' + \
                'Atomic Symbol = (\w+)\n' + \
                '.*\n' + \
                'Relative Atomic Mass = ([\d\.]*).*\n' + \
                '.*\n' + \
                'Standard Atomic Weight = \[?([\d\.]*)(\,([\d\.]+)\])?.*\n'

# [0] = Atomic Symbol; [1] = Relative Atomic Mass;
# [2] = first Standard Weight; [4] = second Standard Weight (if present)
# ignore last digits in ()
list_of_tuples = re.findall( regex_command, text_data )

# dict to get rid of dublicates
average_mass_dict = {}

for tupel in list_of_tuples:
    if tupel[0] in ['D', 'T']:
        # Use Relative Atomic Mass for D and T
        mass = float(tupel[1])
    elif tupel[4] != '':
        # average Standard Weights if two are given
        mass = (float(tupel[2]) + float(tupel[4]))/2
    else:
        mass = float(tupel[2])

    average_mass_dict.update({tupel[0]: np.around(mass,9)})

# write File
cpp_str = """// This script was automatically generated!
// Edit dev/Source/Utils/write_atomic_weight_cpp.py instead!
#ifndef HIPACE_ATOMIC_WEIGHT_TABLE_H_
#define HIPACE_ATOMIC_WEIGHT_TABLE_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include <map>

std::map<std::string, amrex::Real> standard_atomic_weights = {"""

for item in average_mass_dict.items():
    cpp_str += f"""\n    {{"{item[0]}", {item[1]}}},"""

cpp_str += """ };

#endif // #ifndef ATOMIC_WEIGHT_TABLE_H_
"""

f= open('AtomicWeightTable.H','w')
f.write(cpp_str)
f.close()
