# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet
# License: BSD-3-Clause-LBNL

"""
Copyright 2020

This file is part of WarpX.

License: BSD-3-Clause-LBNL
"""
from benchmark import Benchmark
from backend.openpmd_backend import Backend
import sys
import json
import numpy as np

class Checksum:
    '''Class for checksum comparison of one test.
    '''

    def __init__(self, test_name, file_name, do_fields=True, do_particles=True, ndigit=14):
        '''Constructor

        Store test_name and file_name name, and compute checksum
        from file_name and store it in self.data.

        @param self The object pointer.
        @param test_name Name of test, as found between [] in .ini file.
        @param file_name IO file from which the checksum is computed.
        @param do_fields Whether to compare fields in the checksum.
        @param do_particles Whether to compare particles in the checksum.
        @param ndigit Number of digits stored in benchmarks.
        '''

        self.test_name = test_name
        self.file_name = file_name
        self.ndigit = ndigit
        self.data = self.read_output_file(do_fields=do_fields,
                                   do_particles=do_particles)

    def read_output_file(self, do_fields=True, do_particles=True):
        '''Get checksum from IO file.

        Read an simulation IO file, compute 1 checksum per field and return
        all checksums in a dictionary.
        The checksum of quantity Q is max(abs(Q)).

        @param self The object pointer.
        @param do_fields Whether to read fields from the IO file.
        @param do_particles Whether to read particles from the IO file.
        '''

        ds = Backend(self.file_name)
        grid_fields = ds.fields_list()
        species_list = ds.species_list()

        data = {}

        # Compute checksum for field quantities
        if do_fields:
            for lev in range(ds.n_levels()):
                data_lev = {}
                for field in grid_fields:
                    data_lev[field] = self.trim_digits(ds.get_field_checksum(lev, field, self.test_name))
                data['lev=' + str(lev)] = data_lev

        # Compute checksum for particle quantities
        if do_particles and species_list is not None:
            for species in species_list:
                part_fields = ds.get_species_attributes(species)
                data_species = {}
                for field in part_fields:
                    data_species[field] = self.trim_digits(ds.get_species_checksum(species, field))
                data[species] = data_species

        return data

    def evaluate(self, rtol=1.e-9, atol=1.e-40, skip_dict={}):
        '''Compare IO file checksum with benchmark.

        Read checksum from IO file, read benchmark
        corresponding to test_name, and assert that they are equal.
        Almost all the body of this functions is for
        user-readable print statements.

        @param self The object pointer.
        @param test_name Name of test, as found between [] in .ini file.
        @param file_name IO file from which the checksum is computed.
        '''

        print("Checksum evaluation started...")

        ref_benchmark = Benchmark(self.test_name)

        # Dictionaries have same outer keys (levels, species)?
        if (self.data.keys() != ref_benchmark.data.keys()):
            print("ERROR: Benchmark and IO file checksum "
                  "have different outer keys:")
            print("Benchmark: %s" % ref_benchmark.data.keys())
            print("IO file  : %s" % self.data.keys())
            print("\n\n----------------\n\nNew file:")
            print(json.dumps(self.data, indent=2))
            print("\n\n----------------\n")
            sys.exit(1)

        # Dictionaries have same inner keys (field and particle quantities)?
        for key1 in ref_benchmark.data.keys():
            if (self.data[key1].keys() != ref_benchmark.data[key1].keys()):
                print("ERROR: Benchmark and IO file checksum have "
                      "different inner keys:")
                print("Common outer keys: %s" % ref_benchmark.data.keys())
                print("Benchmark inner keys in %s: %s"
                      % (key1, ref_benchmark.data[key1].keys()))
                print("IO file   inner keys in %s: %s"
                      % (key1, self.data[key1].keys()))
                print("\n\n----------------\n\nNew file:")
                print(json.dumps(self.data, indent=2))
                print("\n\n----------------\n")
                sys.exit(1)

        # Dictionaries have same values?
        checksums_differ = False
        for key1 in ref_benchmark.data.keys():
            for key2 in ref_benchmark.data[key1].keys():
                if key1 in skip_dict.keys() and key2 in skip_dict[key1]:
                    continue
                passed = np.isclose(self.data[key1][key2],
                                    ref_benchmark.data[key1][key2],
                                    rtol=rtol, atol=atol)
                if not passed:
                    print("ERROR: Benchmark and IO file checksum have "
                          "different value for key [%s,%s]" % (key1, key2))
                    print("Benchmark: [%s,%s] %.40f"
                          % (key1, key2, ref_benchmark.data[key1][key2]))
                    print("IO file  : [%s,%s] %.40f"
                          % (key1, key2, self.data[key1][key2]))
                    checksums_differ = True
        if checksums_differ:
            print("\n\n----------------\n\nNew file:")
            print(json.dumps(self.data, indent=2))
            print("\n\n----------------\n")
            sys.exit(1)
        print("Checksum evaluation passed.")

    def trim_digits(self, x):
        '''
        Trim the number of signigicant digits

        @param x number to trim
        '''

        if (isinstance(x, int)):
            return x
        else:
            ndigit = self.ndigit
            return float('%s' % float(f'%.{ndigit}g' % x))
