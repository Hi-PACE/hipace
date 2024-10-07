#! /usr/bin/env bash

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, MaxThevenet
# License: BSD-3-Clause-LBNL

# This script runs a simulation in the blowout regime and
# compares the result of the simulation to a benchmark.

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
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        hipace.tile_size = 8 \
        hipace.file_prefix=${TEST_NAME} \
        amr.n_cell = 16 16 32 \
        hipace.dt = 0 \
        geometry.prob_lo = -80.e-6 -80.e-6 -80.e-6 \
        geometry.prob_hi =  80.e-6  80.e-6  80.e-6 \
        beams.names = beam \
        beam.injection_type = from_file \
        beam.input_file = beam_%T.h5 \
        beam.iteration = 0 \
        beam.openPMD_species_name = Electrons

# Compare the beams
$HIPACE_EXAMPLE_DIR/analysis_from_file.py --beam-py beam_%T.h5 \
                                          --beam-out1 ${TEST_NAME}/openpmd_%T.h5 \
                                          --SI
