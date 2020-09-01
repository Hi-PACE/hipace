#! /usr/bin/env sh

# This file is part of the Hipace test suite.
# It runs a Hipace simulation for a can beam with per-slice beam deposition,
# and compares the result of the simulation to a benchmark.

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/can_beam
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs hipace.slice_deposition=1

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --plotfile plt00001 \
    --test-name slice_deposition.1Rank
