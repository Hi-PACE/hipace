#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation in the linear regime with a Gaussian drive beam
# and compares the result with theory.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/linear_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
            beam.profile = gaussian \
            beam.zmin = -59.e-6 \
            beam.zmax = 59.e-6 \
            beam.radius = 100.e-6 \
            beam.mean = 0. 0. 0 \
            beam.std = 20.e-6 20.e-6 14.1e-6 \
            geometry.prob_lo     = -100.e-6   -100.e-6   -60.e-6 \
            geometry.prob_hi     =  100.e-6    100.e-6    60.e-6

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --gaussian-beam

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --plotfile plt00001 \
    --test-name gaussian_linear_wake.SI.1Rank
