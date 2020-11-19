#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation for a can beam in vacuum in serial and parallel and
# checks that they give the same result

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/linear_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -rf plt* REF*

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized

mv plt00001 REF_plt00001

mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized

$HIPACE_EXAMPLE_DIR/analysis_2ranks.py
