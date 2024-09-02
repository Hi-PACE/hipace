#! /usr/bin/env bash

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation of a laser propagating in vacuum
# and compares width and a0 with theory

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/laser
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Relative tolerance for checksum tests depends on the platform
RTOL=1e-12 && [[ "$HIPACE_EXECUTABLE" == *"hipace"*".CUDA."* ]] && RTOL=1e-7

# Run the simulation with multigrid Poisson solver
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        lasers.solver_type = multigrid \
        hipace.file_prefix = $TEST_NAME
# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_vacuum.py --output-dir=$TEST_NAME

rm -rf $TEST_NAME

# Run the simulation with FFT Poisson solver
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        lasers.solver_type = fft \
        hipace.file_prefix = $TEST_NAME
# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_vacuum.py --output-dir=$TEST_NAME
# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --skip-particles \
    --evaluate \
    --rtol $RTOL \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME

rm -rf $TEST_NAME
