#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a small simulation with a fixed ppc beam and a grid current and ensures that they cancel
# each other

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        amr.n_cell = 32 32 32 \
        max_step = 1 \
        hipace.depos_order_xy = 0 \
        geometry.prob_lo = -8. -8. -6. \
        geometry.prob_hi =  8.  8.  6. \
        grid_current.use_grid_current = 1 \
        grid_current.peak_current_density= 0.2 \
        grid_current.position_mean = 0. 0. 0. \
        grid_current.position_std = 0.3 0.3 1.41 \
        hipace.output_period = 1 \
        beam.profile = gaussian \
        beam.position_std = 0.3 0.3 1.41 \
        beam.density = 0.2 \
        beam.radius = 1. \
        beam.ppc = 1 1 1 \

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_grid_current.py

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name diags/h5 \
    --test-name grid_current.1Rank
