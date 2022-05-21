#! /usr/bin/env bash

# Copyright 2020-2022
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs, AlexanderSinn
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation in the linear regime with ion motion
# and compares the result between the two solvers.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/linear_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        random_seed = 1 \
        amr.n_cell = 64 64 200 \
        hipace.bxby_solver = predictor-corrector \
        hipace.normalized_units = 0 \
        hipace.predcorr_B_mixing_factor = 0.0635 \
        hipace.predcorr_max_iterations = 7 \
        hipace.predcorr_B_error_tolerance = 0.0001 \
        plasmas.sort_bin_size = 8 \
        beams.names = beam \
        beam.injection_type = fixed_weight \
        beam.do_symmetrize = 0 \
        beam.num_particles = 1000000 \
        beam.density = ne \
        beam.mass = m_e \
        beam.charge = -q_e \
        beam.u_mean = 10 20 100 \
        beam.u_std = 0 0 0 \
        beam.profile = gaussian \
        beam.position_mean = 0.25*kp_inv 0 2*kp_inv \
        beam.position_std = 0.4*kp_inv 0.4*kp_inv 1.41*kp_inv \
        beam.dy_per_dzeta = 0.2 \
        geometry.prob_lo     = -8*kp_inv   -8*kp_inv   -6*kp_inv \
        geometry.prob_hi     =  8*kp_inv    8*kp_inv    6*kp_inv \
        plasmas.names = elec ions \
        elec.mass = m_e \
        elec.charge = -q_e \
        elec.neutralize_background = false \
        "elec.density(x,y,z)" = ne \
        elec.ppc = 1 1 \
        ions.mass = 5*m_e \
        ions.charge = q_e \
        ions.neutralize_background = false \
        "ions.density(x,y,z)" = ne \
        ions.ppc = 1 1 \
        hipace.file_prefix=$TEST_NAME/pc

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        random_seed = 1 \
        amr.n_cell = 64 64 200 \
        hipace.bxby_solver = explicit \
        hipace.normalized_units = 0 \
        plasmas.sort_bin_size = 8 \
        beams.names = beam \
        beam.injection_type = fixed_weight \
        beam.do_symmetrize = 0 \
        beam.num_particles = 1000000 \
        beam.density = ne \
        beam.mass = m_e \
        beam.charge = -q_e \
        beam.u_mean = 10 20 100 \
        beam.u_std = 0 0 0 \
        beam.profile = gaussian \
        beam.position_mean = 0.25*kp_inv 0 2*kp_inv \
        beam.position_std = 0.4*kp_inv 0.4*kp_inv 1.41*kp_inv \
        beam.dy_per_dzeta = 0.2 \
        geometry.prob_lo     = -8*kp_inv   -8*kp_inv   -6*kp_inv \
        geometry.prob_hi     =  8*kp_inv    8*kp_inv    6*kp_inv \
        plasmas.names = elec ions \
        elec.mass = m_e \
        elec.charge = -q_e \
        elec.neutralize_background = false \
        "elec.density(x,y,z)" = ne \
        elec.ppc = 1 1 \
        ions.mass = 5*m_e \
        ions.charge = q_e \
        ions.neutralize_background = false \
        "ions.density(x,y,z)" = ne \
        ions.ppc = 1 1 \
        hipace.file_prefix=$TEST_NAME/e

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_equal.py --first=$TEST_NAME/pc  --second=$TEST_NAME/e

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME/e \
    --test-name $TEST_NAME
