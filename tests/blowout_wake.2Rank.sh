#! /usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Andrew Myers, Axel Huebl, MaxThevenet, Severin Diederichs
#
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation in the blowout regime and compares the result
# with SI units.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

echo $HIPACE_EXECUTABLE

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Relative tolerance for checksum tests depends on the platform
RTOL=1e-12 && [[ "$HIPACE_EXECUTABLE" == *"hipace"*".CUDA."* ]] && RTOL=2e-5

rm -rf si_data
rm -rf si_data_fixed_weight
rm -rf normalized_data
rm -rf normalized_data_cd2
# Run the simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        plasmas.sort_bin_size = 8 \
        hipace.file_prefix=si_data/ \
        max_step=1

mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        plasmas.sort_bin_size = 8 \
        beam.injection_type=fixed_weight \
        beam.num_particles=1000000 \
        hipace.file_prefix=si_data_fixed_weight/ \
        max_step=1

mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        plasmas.sort_bin_size = 8 \
        hipace.file_prefix=normalized_data/ \
        max_step=1

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py \
    --normalized-data normalized_data/ \
    --si-data si_data/ \
    --si-fixed-weight-data si_data_fixed_weight/

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol $RTOL \
    --file_name normalized_data/ \
    --test-name blowout_wake.2Rank \
    --skip "{'beam': 'id'}"

echo "Start testing plasma reordering"

mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        plasmas.sort_bin_size = 8 \
        hipace.file_prefix=normalized_data_cd2/ \
        plasmas.reorder_period = 4 \
        max_step=1

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol $RTOL \
    --file_name normalized_data_cd2/ \
    --test-name blowout_wake.2Rank \
    --skip "{'beam': 'id'}"
