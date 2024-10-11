#! /usr/bin/env bash

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, MaxThevenet
# License: BSD-3-Clause-LBNL

# This file runs a Hipace simulation in the blowout regime and cand compares the result
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

# Run the simulation
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        hipace.file_prefix=${TEST_NAME}_1 \
        amr.n_cell = 16 16 32 \
        geometry.prob_lo = -2. -2. -12. \
        geometry.prob_hi =  2.  2.  12. \
        hipace.dt = 0

# Restart the simulation with previous beam output
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        hipace.file_prefix=${TEST_NAME}_2 \
        amr.n_cell = 24 24 48 \
        geometry.prob_lo = -2. -2. -12. \
        geometry.prob_hi =  2.  2.  12. \
        hipace.dt = 0 \
        beam.injection_type = from_file \
        beam.input_file = ${TEST_NAME}_1/openpmd_%T.h5 \
        beam.iteration = 0 \
        beam.openPMD_species_name = beam \
        beam.plasma_density = 0 # use value from file

# Compare the beams
$HIPACE_EXAMPLE_DIR/analysis_from_file.py --beam-out1 ${TEST_NAME}_1/openpmd_%T.h5 \
                                          --beam-out2 ${TEST_NAME}_2/openpmd_%T.h5
