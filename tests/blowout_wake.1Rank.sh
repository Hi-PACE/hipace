#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation in the blowout regime and compares the result
# with SI units.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -rf si_data
rm -rf plt00001
# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI
mv plt00001 si_data
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized
mv plt00001 normalized_data

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --normalized-data normalized_data --si-data si_data

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --plotfile normalized_data \
    --test-name blowout_wake.1Rank
