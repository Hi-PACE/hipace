#! /usr/bin/env bash

# This file is part of the HiPACE++ test suite.
# It checks that a scaled-down version of production LWFA and PWFA simulations run correctly.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/get_started
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the PWFA test and verify checksum
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_pwfa \
        max_step = 10 \
        amr.n_cell = 64 64 100 \
        hipace.file_prefix = ${TEST_NAME}_pwfa

$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name ${TEST_NAME}_pwfa \
    --test-name ${TEST_NAME}_pwfa

# Run the LWFA test and verify checksum
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_lwfa \
        max_step = 10 \
        amr.n_cell = 64 64 100 \
        hipace.file_prefix = ${TEST_NAME}_lwfa

$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name ${TEST_NAME}_lwfa \
    --test-name ${TEST_NAME}_lwfa

# Compare the results with checksum benchmark
#     --skip-particles \
#     --skip "{'lev=0' : ['Sy', 'Sx', 'chi']}"
