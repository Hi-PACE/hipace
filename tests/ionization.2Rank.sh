#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation with in neutral hydrogen that gets ionized by the beam

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -rf ionization_data

# Run the simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_ionization_SI \
        hipace.dt = 1e-12 \
        hipace.output_period = 2 \
        hipace.file_prefix=ionization_data/ \
        max_step=2

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name ionization_data/ \
    --test-name ionization.2Rank \
    --skip "{'beam': 'id'}"
