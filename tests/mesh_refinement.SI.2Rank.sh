#! /usr/bin/env bash

# This file is part of the HiPACE++ test suite.
# It checks that a scaled-down version of mesh refinement simulations run correctly.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/get_started
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the mesh refinement test and verify checksum
mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_mesh_refinement \
        max_step = 1 \
        amr.n_cell = 63 63 100 \
        mr_lev1.n_cell = 31 31 \
        driver.num_particles = 1e6 \
        witness.num_particles = 1e6 \
        hipace.depos_order_xy = 1 \
        hipace.file_prefix = $TEST_NAME

$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
