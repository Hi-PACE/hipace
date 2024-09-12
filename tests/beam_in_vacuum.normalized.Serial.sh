#! /usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation for a can beam in vacuum, and compares the result
# of the simulation to theory.

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
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
                   hipace.tile_size = 8 \
                   hipace.depos_order_xy=0 \
                   diagnostic.field_data = all rho \
                   hipace.file_prefix=$TEST_NAME

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --normalized-units --output-dir=$TEST_NAME

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
