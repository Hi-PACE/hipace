#! /usr/bin/env bash

# Copyright 2025
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, MaxThevenet
# License: BSD-3-Clause-LBNL

# This script runs a simulation in the blowout regime and
# compares the result of the simulation to a benchmark.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# gererate beam
python3 ${HIPACE_SOURCE_DIR}/tools/write_plasma_density.py

# Run the simulation
$HIPACE_EXECUTABLE \
        hipace.file_prefix=${TEST_NAME} \
        amr.n_cell = 31 31 31 \
        my_constants.ne = 1e24 \
        my_constants.channel_radius = 40e-6 \
        my_constants.ramp_length = 60e-6 \
        my_constants.wp = "sqrt(ne * q_e^2  / (epsilon0 * m_e))" \
        my_constants.kp = wp / clight \
        my_constants.kp_inv = 1. / kp \
        amr.max_level = 0 \
        max_step = 1 \
        hipace.dt = "500e-6 / clight / 20" \
        hipace.verbose = 3 \
        hipace.depos_order_xy = 2 \
        boundary.field = Dirichlet \
        boundary.particle = Absorbing \
        geometry.prob_lo = -50e-6  -50e-6  -50e-6 \
        geometry.prob_hi =  50e-6   50e-6   50e-6 \
        plasmas.names = elec1 elec2 \
        plasmas.neutralize_background = true \
        'elec1.density(x,y,z)' = '"ne * (1 + (x^2 + y^2) / channel_radius^2) \
                                  * if(z < ramp_length, z / ramp_length, 1)"' \
        elec2.read_density_from_path = "example-density.h5" \
        elec1.ppc = 1 1 \
        elec1.element = electron \
        elec2.ppc = 1 1 \
        elec2.element = electron \
        diagnostic.output_period = 1 \
        diagnostic.diag_type = xyz \
        diagnostic.field_data = rho_elec \
        diagnostic.patch_lo = "-6*kp_inv"  "-6*kp_inv"  0 \
        diagnostic.patch_hi =  "6*kp_inv"   "6*kp_inv"  0 \

$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
