#! /usr/bin/env bash

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation with a beam in vacuum with transverse currents
# in serial and parallel and asserts that both give the same result.
# This checks the complex communication patterns to handle next slice of slice 0
# of each box.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -rf serial
rm -rf parallel

# Run the serial simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized_transverse \
        hipace.tile_size = 8 \
        hipace.file_prefix=serial/

# Run the parallel simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized_transverse \
        hipace.tile_size = 8 \
        hipace.file_prefix=parallel/

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_transverse.py \
    --serial serial/ \
    --parallel parallel/
