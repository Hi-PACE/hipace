#! /usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Andrew Myers, Axel Huebl, MaxThevenet, Severin Diederichs
#
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation for a can beam, and compares the result
# of the simulation to a benchmark.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Relative tolerance for checksum tests depends on the platform
RTOL=1e-12 && [[ "$HIPACE_EXECUTABLE" == *"hipace"*".CUDA."* ]] && RTOL=1.3e-5

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized max_step=2 \
        hipace.tile_size = 8 \
        hipace.MG_tolerance_rel = 1e-5 \
        hipace.file_prefix=$TEST_NAME

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol $RTOL \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME \
    --skip "{'beam': 'id'}"
