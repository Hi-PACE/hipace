#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation in the linear regime and compares the result
# with theory.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/linear_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
            beam.profile = gaussian \
            beam.zmin = -5.9 \
            beam.zmax = 5.9 \
            beam.radius = 10 \
            beam.position_mean = 0. 0. 0 \
            beam.position_std = 2 2 1.41 \
            geometry.prob_lo     = -10.   -10.   -6  \
            geometry.prob_hi     =  10.    10.    6

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --normalized-units --gaussian-beam

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --plotfile plt00001 \
    --test-name gaussian_linear_wake.normalized.1Rank
