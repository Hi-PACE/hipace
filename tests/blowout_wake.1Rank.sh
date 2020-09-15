#! /usr/bin/env sh

# This file is part of the Hipace test suite.
# It runs a Hipace simulation in the blowout regime and compares the result
# with SI units.

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -r si_data
rm -r plt00001
# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_si
mv plt00001 si_data
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized
#mv plt00001 normalized_data

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --normalized-data plt00001 --si-data si_data

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --plotfile plt00001 \
    --test-name blowout_wake.1Rank
