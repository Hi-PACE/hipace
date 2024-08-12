#! /usr/bin/env bash

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: Xingjian Hui
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


# Run the simulation with initial phi2
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        laser.phi2 = 2.4e-26 \
        hipace.file_prefix = $TEST_NAME
        laser.w0 = 30e-6
        laser.L0 = 5e-6
# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_init_chirp.py --output-dir=$TEST_NAME \
        --chirp_type = phi2

rm -rf $TEST_NAME

# Run the simulation with initial zeta
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        laser.zeta = 2.4e-26 \
        hipace.file_prefix = $TEST_NAME
        laser.w0 = 30e-6
        laser.L0 = 5e-6
# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_init_chirp.py --output-dir=$TEST_NAME
        --chirp_type = zeta

rm -rf $TEST_NAME