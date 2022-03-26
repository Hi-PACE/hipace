#! /usr/bin/env bash

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation for a can beam in vacuum with open boundary conditions,
# and compares the result of the simulation to theory.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        plasmas.sort_bin_size = 8 \
        hipace.depos_order_xy=0 \
        geometry.is_periodic = false false false \
        fields.extended_solve = true \
        fields.open_boundary = true \
        geometry.prob_lo     = -4.   -4.   -2.  \
        geometry.prob_hi     =  4.    4.    2.  \
        beam.position_mean = 2. -1. 0. \
        hipace.file_prefix=$TEST_NAME

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_open_boundary.py --normalized-units --output-dir=$TEST_NAME

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
