#! /usr/bin/env bash

# Copyright 2025
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, MaxThevenet
# License: BSD-3-Clause-LBNL

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/plasma_ionization
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests


# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_plasma_ionization \
        hipace.file_prefix=grid_ionization.1Rank \
        grid_ionization.plasma_names=ion\

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_plasma_ionization \
        hipace.file_prefix=particle_ionization.1Rank \
        ion.ppc=10 10\
        hipace.deposit_n=1\
        hipace.deposit_n_ion_levels=1



python3 $HIPACE_EXAMPLE_DIR/analysis_plasma_ionization.py \
    --diags_grid=$HIPACE_EXAMPLE_DIR/grid_ionization.1Rank \
    --diags_particle=$HIPACE_EXAMPLE_DIR/particle_ionization.1Rank
