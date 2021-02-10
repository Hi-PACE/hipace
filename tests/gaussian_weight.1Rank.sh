#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation initializing a Gaussian beam, and compares the result
# of the simulation to theory.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/gaussian_weight
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
                    beam.dx_per_dzeta=0.1 \
                    beam.dy_per_dzeta=-0.2 \
                    beam.position_std = 0.1 0.1 2.

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --normalized-units --tilted-beam

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --normalized-units

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name diags/h5 \
    --rtol=.01 \
    --test-name gaussian_weight.1Rank
