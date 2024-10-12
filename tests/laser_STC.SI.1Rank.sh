#! /usr/bin/env bash

# Copyright 2024
#
# This file is part of HiPACE++.
#
# Authors: Xingjian Hui
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It initializes a Hipace simulation of a gaussian laser with initial STC parameters
# and test the correctness of the STC factors

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/laser
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the simulation with initial phi2
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_chirp \
        laser.phi2 = 2.4e-26 \
        laser.chirp_theta_xy = 0 \
        hipace.file_prefix = $TEST_NAME \
# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_init_chirp.py --output-dir=$TEST_NAME \
        --chirp_type="phi2"

rm -rf $TEST_NAME

# Run the simulation with initial zeta
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_chirp \
        laser.zeta = 2.4e-19 \
        laser.chirp_theta_xy = 0 \
        hipace.file_prefix = $TEST_NAME \

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_init_chirp.py --output-dir=$TEST_NAME \
        --chirp_type="zeta"

rm -rf $TEST_NAME

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_chirp \
        laser.beta= 2e-17 \
        laser.chirp_theta_xy = 0 \
        hipace.file_prefix = $TEST_NAME \

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_init_chirp.py --output-dir=$TEST_NAME \
        --chirp_type="beta"

rm -rf $TEST_NAME
