#! /usr/bin/env bash

# Copyright 2026
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
#
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

echo $HIPACE_EXECUTABLE

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Relative tolerance for checksum tests depends on the platform
RTOL=1e-12

rm -rf initial_sim
rm -rf restart_sim

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_plasma_restart_1_SI \
        hipace.tile_size = 8 \
        hipace.file_prefix = "initial_sim/"

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_plasma_restart_2_SI \
        hipace.tile_size = 8 \
        hipace.file_prefix = "restart_sim/"

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_plasma_restart.py

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol $RTOL \
    --file_name restart_sim/ \
    --test-name plasma_restart.SI.1Rank \
    --skip "{'beam': 'id'}"
