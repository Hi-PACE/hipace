#! /usr/bin/env bash

# Copyright 2021
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation in the blowout regime and compares the result
# with SI units.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -rf hosing_data

# Run the simulation
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        hipace.dt = 20 \
        diagnostic.output_period = 10 \
        beam.injection_type = fixed_weight \
        beam.num_particles = 1000000 \
        beam.density = 200 \
        beam.position_std = 0.1 0.1 1.41 \
        beam.dx_per_dzeta = 0.2 \
        plasmas.names = plasma ions \
        plasma.neutralize_background = 0 \
        "ions.density(x,y,z)" = 1. \
        ions.ppc = 1 1 \
        ions.charge = 1 \
        ions.mass = 1836 \
        ions.neutralize_background = 0 \
        hipace.file_prefix=hosing_data/ \
        max_step=10

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name hosing_data/ \
    --test-name hosing.2Rank \
    --skip "{'beam': 'id'}"
