#! /usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation with full IO and xz-slice IO, and checks that the results are equal.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        diagnostic.diag_type=xyz \
        amr.n_cell = 64 88 100 \
        hipace.file_prefix=full_io

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        diagnostic.diag_type=xz \
        amr.n_cell = 64 88 100 \
        hipace.file_prefix=slice_io_xz

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        diagnostic.diag_type=yz \
        amr.n_cell = 64 88 100 \
        hipace.file_prefix=slice_io_yz

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        diagnostic.diag_type=xyz \
        amr.n_cell = 64 88 100 \
        diagnostic.patch_lo = -3 -100  0 \
        diagnostic.patch_hi =  3  100  0 \
        hipace.file_prefix=slice_io_cut_xy

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        hipace.tile_size = 8 \
        diagnostic.diag_type=xz \
        amr.n_cell = 64 88 100 \
        diagnostic.patch_lo =  0 -3 -10 \
        diagnostic.patch_hi =  4  3  10 \
        hipace.file_prefix=slice_io_cut_xz

# assert whether the two IO types match
$HIPACE_EXAMPLE_DIR/analysis_slice_IO.py
