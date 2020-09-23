#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation in the blowout regime and compares the result
# with SI units.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/gaussian_blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --plotfile plt00001 \
    --test-name gaussian_blowout_wake.1Rank
