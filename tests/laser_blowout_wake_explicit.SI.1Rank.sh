#! /usr/bin/env bash

# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation in normalized units
# in the blowout regime of a laser driver and compares the checksum with a benchmark.
# This case was benchmarked against INF&RNO simulations and therefore a change in the benchmark
# should be accepted only with great care!

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        hipace.tile_size = 8 \
        hipace.file_prefix=$TEST_NAME \
        max_step = 0 \
        beams.names = no_beam \
        geometry.prob_lo     = -20.*kp_inv   -20.*kp_inv   -7.5*kp_inv  \
        geometry.prob_hi     =  20.*kp_inv    20.*kp_inv    6.*kp_inv  \
        lasers.names = laser \
        lasers.lambda0 = .8e-6 \
        laser.a0 = 4.5 \
        laser.position_mean = 0. 0. 0 \
        laser.w0 = 4.*kp_inv \
        laser.L0 = 2.*kp_inv \
        amr.n_cell = 128 128 100 \

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME \
    --skip-particles \
    --skip "{'lev=0' : ['Sy', 'Sx', 'chi']}"
