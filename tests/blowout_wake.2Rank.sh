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
rm -rf si_data_fixed_weight
rm -rf normalized_data
# Run the simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        hipace.file_prefix=si_data/ \
        hipace.verbose=2 \
        max_step=2

mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        beam.injection_type=fixed_weight \
        hipace.verbose=2 \
        beam.num_particles=1000000 \
        hipace.file_prefix=si_data_fixed_weight/ \
        max_step=2

mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.file_prefix=normalized_data/ \
        hipace.verbose=2 \
        max_step=2

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py \
    --normalized-data normalized_data/ \
    --si-data si_data/ \
    --si-fixed-weight-data si_data_fixed_weight/

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name normalized_data/ \
    --test-name blowout_wake.2Rank \
    --skip "{'beam': 'id'}"
