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

rm -rf plt00001
# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized hipace.output_slice=0
rm -rf full_io
mv plt00001 full_io
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized hipace.output_slice=1
rm -rf slice_io
mv plt00001 slice_io

# assert whether the two IO types match
$HIPACE_EXAMPLE_DIR/analysis_slice_IO.py

# Make sure that the slice is much smaller than the full diagnostics
size_full=$(du -s full_io/Level_0 | cut -f1)
size_slice=$(du -s slice_io/Level_0 | cut -f1)

if [[ $((size_full/size_slice<120)) == 1 ]]; then
    echo $size_full
    echo $size_slice
    echo "ERROR: field data should be ~128x smaller for slice than for full diagnostics"
    exit
fi
