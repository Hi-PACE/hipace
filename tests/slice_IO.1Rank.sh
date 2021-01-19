#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation with full IO and xz-slice IO, and checks that the results are equal.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -rf diags
# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        diagnostic.diag_type=xyz \
        amr.n_cell = 64 86 100
rm -rf full_io
mv diags/h5 full_io

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        diagnostic.diag_type=xz \
        amr.n_cell = 64 86 100
rm -rf slice_io_xz
mv diags/h5 slice_io_xz

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        diagnostic.diag_type=yz \
        amr.n_cell = 64 86 100
rm -rf slice_io_yz
mv diags/h5 slice_io_yz

# assert whether the two IO types match
$HIPACE_EXAMPLE_DIR/analysis_slice_IO.py
