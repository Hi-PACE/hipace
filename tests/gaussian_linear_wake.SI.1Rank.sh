#! /usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation in the linear regime with a Gaussian drive beam
# and compares the result with theory.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/linear_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Relative tolerance for checksum tests depends on the platform
RTOL=1e-11 && [[ "$HIPACE_EXECUTABLE" == *"hipace"*".CUDA."* ]] && RTOL=1e-7

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        hipace.tile_size = 8 \
        beam.profile = gaussian \
        beam.zmin = -59.e-6 \
        beam.zmax = 59.e-6 \
        beam.radius = 100.e-6 \
        beam.position_mean = 0. 0. 0 \
        beam.position_std = 20.e-6 20.e-6 14.1e-6 \
        geometry.prob_lo     = -100.e-6   -100.e-6   -60.e-6 \
        geometry.prob_hi     =  100.e-6    100.e-6    60.e-6 \
        diagnostic.field_data = all rho \
        hipace.file_prefix=$TEST_NAME

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --gaussian-beam --output-dir=$TEST_NAME

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol $RTOL \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
