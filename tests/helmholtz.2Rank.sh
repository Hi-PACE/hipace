#! /usr/bin/env bash

# Copyright 2020-2026
#
# This file is part of HiPACE++.
#
# Authors: Andrew Myers, Axel Huebl, MaxThevenet, Severin Diederichs, Alexander Sinn
#
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation in the blowout regime and compares the result
# with SI units.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/helmholtz
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

rm -rf $TEST_NAME

### AVERAGED ###
################

mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs \
        hipace.output_folder = $TEST_NAME

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --mode='averaged' --output-dir=$TEST_NAME

rm -rf $TEST_NAME

### UNAVERAGED ###
##################


 mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs \
         my_constants.nsteps_per_period = 8 \
         my_constants.dz_per_wavelength = 8 \
         beam.n_subcycles = 1 \
         helmholtz.mode = FullWave \
         my_constants.part_per_slice = 128 \
         my_constants.Fseed = Eseed \
         my_constants.nx=32 \
         max_step="nsteps_per_period*(Q1+lq/2)/lu" \
         hipace.output_folder = $TEST_NAME

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis.py --mode='unaveraged' --output-dir=$TEST_NAME

rm -rf $TEST_NAME
