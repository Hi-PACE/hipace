#! /usr/bin/env bash

# Copyright 2026
#
# This file is part of HiPACE++.
#
# Authors: Xingjian Hui, AlexanderSinn, MaxThevenet
# License: BSD-3-Clause-LBNL

# abort on first encounted error
set -eu -o pipefail

HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/plasma_ionization

GRID_DIAGS="${PWD}/grid_ionization.1Rank"
PARTICLE_DIAGS="${PWD}/particle_ionization.1Rank"

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_plasma_ionization \
    hipace.file_prefix=$GRID_DIAGS \
    ion.ppc=0 0 \
    grid_ionization.plasma_names=ion

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_plasma_ionization \
    hipace.file_prefix=$PARTICLE_DIAGS \
    hipace.deposit_n=1 \
    hipace.deposit_n_ion_levels=1

python3 $HIPACE_EXAMPLE_DIR/analysis_plasma_ionization.py \
    --diags_grid=$GRID_DIAGS \
    --diags_particle=$PARTICLE_DIAGS
