# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL

import re
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

class Backend:
    ''' Use openPMD-viewer as the backend reader to read openPMD files
    '''

    def __init__(self, filename):
        ''' Constructor: store the dataset object
        '''

        self.dataset = OpenPMDTimeSeries(filename, backend='h5py')

    def fields_list(self):
        ''' Return the list of fields defined on the grid
        '''

        return self.dataset.avail_fields

    def species_list(self):
        ''' Return the list of species in the dataset
        '''

        return self.dataset.avail_species

    def n_levels(self):
        ''' Return the number of MR levels in the dataset
        '''

        return 1

    def get_field_checksum(self, lev, field, test_name):
        ''' Calculate the checksum for a given field at a given level in the dataset
        '''

        Q = self.dataset.get_field(field=field, iteration=self.dataset.iterations[-1])[0]
        return np.sum(np.abs(Q))

    def get_species_attributes(self, species):
        ''' Return the list of attributes for a given species in the dataset
        '''
        return self.dataset.avail_record_components[species]

    def get_species_checksum(self, species, attribute):
        ''' Calculate the checksum for a given attribute of a given species in the dataset
        '''

        Q = self.dataset.get_particle(var_list=[attribute], species=species,
                                      iteration=self.dataset.iterations[-1])
        # JSON complains with numpy integers, so if the quantity is a np.int64, convert to int
        checksum = np.sum(np.abs(Q))
        if type(checksum) in [np.int64, np.uint64]:
            return int(checksum)
        return checksum
