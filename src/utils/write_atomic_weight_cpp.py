#! /usr/bin/env python

"""
This python script reads atomic masses tables in relative_atomic_masses.txt
(generated from the NIST website) and extracts
standard atomic weights into C++ file AtomicWeightTable.H.

To update AtomicWeightTable.H go to http://physics.nist.gov/Comp
select 'All Elements', 'Linearized ASCII Output', 'Most common isotopes',
'Get Data' and then copy the data into a text file in src/utils/ named
'relative_atomic_masses.txt'. Finally, run this script.
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
// Edit src/utils/write_atomic_weight_cpp.py instead!
#ifndef HIPACE_ATOMIC_WEIGHT_TABLE_H_
#define HIPACE_ATOMIC_WEIGHT_TABLE_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include <map>

// Reference:
// Coursey, J.S., Schwab, D.J., Tsai, J.J., and Dragoset, R.A. (2015),
// Atomic Weights and Isotopic Compositions (version 4.1).
// [Online] Available: http://physics.nist.gov/Comp [2021, 05, 19].
// National Institute of Standards and Technology, Gaithersburg, MD.
//
// The Data written below is a reformatting of the data referenced form NIST.

std::map<std::string, amrex::Real> standard_atomic_weights = {"""

for item in average_mass_dict.items():
    cpp_str += f"""\n    {{"{item[0]}", {item[1]}}},"""

cpp_str += """ };

#endif // #ifndef ATOMIC_WEIGHT_TABLE_H_
"""

with open('AtomicWeightTable.H','w') as f:
    f.write(cpp_str)
