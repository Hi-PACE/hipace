#! /usr/bin/env bash

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, MaxThevenet
# License: BSD-3-Clause-LBNL


# This script runs a Hipace simulation in the blowout regime and cand compares the result
# of the simulation to a benchmark.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# gererate beam
python3 ${HIPACE_SOURCE_DIR}/tools/write_beam.py

# Run the simulation
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        hipace.file_prefix=${TEST_NAME} \
        amr.n_cell = 16 16 32 \
        hipace.dt = 0 \
        geometry.prob_lo = -8. -8. -8. \
        geometry.prob_hi =  8.  8.  8. \
        beam.injection_type = from_file \
        beam.input_file = beam_%T.h5 \
        beam.iteration = 0 \
        beam.openPMD_species_name = Electrons \
        beam.plasma_density = 2.8239587008591567e23 # to convert beam to normalized units

# Compare the beams
$HIPACE_EXAMPLE_DIR/analysis_from_file.py --beam-py beam_%T.h5 \
                                          --beam-out1 ${TEST_NAME}/openpmd_%T.h5
