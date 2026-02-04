#! /usr/bin/env bash

# This file is part of the HiPACE++ test suite.
# It checks that a scaled-down version of production PWFA simulations run correctly with SALAME.

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
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_pwfa \
        max_step = 0 \
        amr.n_cell = 64 64 100 \
        witness.do_salame = 1 \
        witness.profile = can \
        witness.zmin = -150e-6 \
        witness.zmax = -100e-6 \
        hipace.file_prefix = $TEST_NAME

$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol=5.e-6 \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
