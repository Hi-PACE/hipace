#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation in the blowout regime and cand compares the result
# of the simulation to a benchmark.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/from_file
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# gererate beam
python3 ${HIPACE_SOURCE_DIR}/tools/write_beam.py

# Run the simulation
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        hipace.file_prefix=${TEST_NAME}_1 \
        hipace.dt = 0 \
        max_step = 0 \
        plasmas.names = no_plasma

# Restart the simulation with previous beam output
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        hipace.file_prefix=${TEST_NAME}_2 \
        hipace.dt = 0 \
        max_step = 0 \
        plasmas.names = no_plasma \
        beam.input_file = ${TEST_NAME}_1/openpmd_%T.h5 \
        beam.openPMD_species_name = beam

# Compare the beams
$HIPACE_EXAMPLE_DIR/analysis.py --beam-py beam_%T.h5 \
                                --beam-out1 ${TEST_NAME}_1/openpmd_%T.h5 \
                                --beam-out2 ${TEST_NAME}_2/openpmd_%T.h5 \
                                --SI
