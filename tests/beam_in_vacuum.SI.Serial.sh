#! /usr/bin/env sh
set -e
# This file is part of the Hipace test suite.
# It runs a Hipace simulation for a can beam in vacuum, and compares the result
# of the simulation to theory.

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Run the simulation
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI hipace.depos_order_xy=0

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --plotfile plt00001 \
    --test-name beam_in_vacuum.SI.Serial
