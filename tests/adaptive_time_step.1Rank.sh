#! /usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a HiPACE++ simulation in the blowout regime and compares the result
# with SI units.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Relative tolerance for checksum tests depends on the platform
RTOL=1e-12 && [[ "$HIPACE_EXECUTABLE" == *"hipace"*".CUDA."* ]] && RTOL=1e-5

rm -rf negative_gradient.txt
rm -rf positive_gradient.txt

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        amr.n_cell = 32 32 32 \
        max_step = 20 \
        geometry.prob_lo = -2. -2. -2. \
        geometry.prob_hi =  2.  2.  2. \
        hipace.dt = 0. \
        diagnostic.output_period = 0 \
        beam.density = 1 \
        beam.radius = 1. \
        beam.n_subcycles = 4 \
        beam.ppc = 4 4 1 \
        'beams.external_E(x,y,z,t) = 0. 0. -.5*z' \
        hipace.verbose=1\
        hipace.dt=adaptive\
        plasmas.adaptive_density=1 \
        hipace.nt_per_betatron = 89.7597901025655 \
        > negative_gradient.txt

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        amr.n_cell = 32 32 32 \
        max_step = 20 \
        geometry.prob_lo = -2. -2. -2. \
        geometry.prob_hi =  2.  2.  2. \
        hipace.dt = 0. \
        diagnostic.output_period = 20 \
        beam.density = 1 \
        beam.radius = 1. \
        beam.n_subcycles = 4 \
        beam.ppc = 4 4 1 \
        'beams.external_E(x,y,z,t) = 0. 0. .5*z' \
        hipace.verbose=1\
        hipace.dt = adaptive\
        plasmas.adaptive_density=1 \
        hipace.nt_per_betatron = 89.7597901025655 \
        hipace.file_prefix=$TEST_NAME \
        > positive_gradient.txt

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_adaptive_ts.py

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol $RTOL \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
