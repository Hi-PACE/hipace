#! /usr/bin/env bash

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation with plasma collisions and compare with benchmark
# in normalized units

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the simulation
OMP_NUM_THREADS=1 mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        hipace.do_tiling = 0 \
        hipace.file_prefix=$TEST_NAME \
        hipace.collisions = collision1 \
        collision1.species = plasma plasma

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME \
    --skip "{'beam': 'id'}"
