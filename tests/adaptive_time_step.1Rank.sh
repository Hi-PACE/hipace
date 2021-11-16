#! /usr/bin/env bash

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

rm -rf negative_gradient_data
rm -rf positive_gradient_data

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        plasmas.sort_bin_size = 8 \
        amr.n_cell = 32 32 32 \
        max_step = 20 \
        geometry.prob_lo = -2. -2. -2. \
        geometry.prob_hi =  2.  2.  2. \
        hipace.dt = 0. \
        hipace.output_period = 20 \
        beam.density = 1 \
        beam.radius = 1. \
        beam.n_subcycles = 4 \
        beam.ppc = 4 4 1 \
        hipace.external_Ez_slope = -.5 \
        hipace.verbose=1\
        hipace.dt=adaptive\
        plasmas.adaptive_density=1 \
        hipace.nt_per_betatron = 89.7597901025655 \
        > negative_gradient.txt

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        plasmas.sort_bin_size = 8 \
        amr.n_cell = 32 32 32 \
        max_step = 20 \
        geometry.prob_lo = -2. -2. -2. \
        geometry.prob_hi =  2.  2.  2. \
        hipace.dt = 0. \
        hipace.output_period = 20 \
        beam.density = 1 \
        beam.radius = 1. \
        beam.n_subcycles = 4 \
        beam.ppc = 4 4 1 \
        hipace.external_Ez_slope = .5 \
        hipace.verbose=1\
        hipace.dt = adaptive\
        plasmas.adaptive_density=1 \
        hipace.nt_per_betatron = 89.7597901025655 \
        > positive_gradient.txt

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_adaptive_ts.py

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name diags/hdf5 \
    --test-name adaptive_time_step.1Rank
