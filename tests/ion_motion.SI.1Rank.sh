#! /usr/bin/env bash

# Copyright 2020-2022
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs, AlexanderSinn
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation in the linear regime with ion motion
# and compares the result between the two solvers.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/linear_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_ion_motion_SI \
        hipace.bxby_solver = predictor-corrector \
        hipace.file_prefix=$TEST_NAME/pc

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_ion_motion_SI \
        hipace.bxby_solver = explicit \
        hipace.file_prefix=$TEST_NAME/e

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_equal.py --first=$TEST_NAME/pc  --second=$TEST_NAME/e

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME/e \
    --test-name $TEST_NAME \
    --skip "{'lev=0' : ['Sy', 'Sx', 'Mult']}"
