#! /usr/bin/env bash

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation with full and coarse IO and compares them

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the fine simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        amr.n_cell = 60 60 100 \
        hipace.tile_size = 16 \
        max_step = 1 \
        diagnostic.field_data = Ez ExmBy EypBx Bx By Bz \
        hipace.file_prefix=fine_io

# Run the coarse simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        amr.n_cell = 60 60 100 \
        hipace.tile_size = 16 \
        max_step = 1 \
        diagnostic.field_data = Ez ExmBy EypBx Bx By Bz \
        diagnostic.coarsening = 3 4 5 \
        hipace.file_prefix=coarse_io

# Compare the results
$HIPACE_EXAMPLE_DIR/analysis_coarsening.py

