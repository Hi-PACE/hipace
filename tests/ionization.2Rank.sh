#! /usr/bin/env bash

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
# Severin Diederichs
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation with in neutral hydrogen that gets ionized by the beam

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

rm -rf $TEST_NAME

# Run the simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_ionization_SI \
        hipace.tile_size = 8 \
        hipace.dt = 1e-12 \
        diagnostic.output_period = 2 \
        hipace.file_prefix=$TEST_NAME \
        max_step=2

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME \
    --skip "{'beam': 'id'}"
