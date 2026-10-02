#! /usr/bin/env bash

# Copyright 2020-2025
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

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/helmholtz
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Relative tolerance for checksum tests depends on the platform
RTOL=1e-12 && [[ "$HIPACE_EXECUTABLE" == *"hipace"*".CUDA."* ]] && RTOL=2e-5

rm -rf helmholtz.1

# Run the simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_1 \
        my_constants.nx = 32 \
        my_constants.Lbeam = 4*lr \
        my_constants.dz_per_wavelength = 5 \
        beam.num_particles = 1e3 \
        beamsub.num_particles = 1e2 \
        hipace.file_prefix = helmholtz.1 \
        max_step=10

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol $RTOL \
    --file_name helmholtz.1 \
    --test-name helmholtz.1

rm -rf helmholtz.1
rm -rf diags
