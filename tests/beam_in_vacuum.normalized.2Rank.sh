#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation for a can beam in vacuum in serial and parallel and
# checks that they give the same result

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -rf plt* REF*

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        amr.n_cell=128 256 30 \
        diagnostic.type = xyz \
        beam.radius = 20.

mv plt00001 REF_plt00001

mpiexec -n 2 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        amr.n_cell=128 256 30 \
        diagnostic.type = xyz \
        beam.radius = 20.

$HIPACE_EXAMPLE_DIR/analysis_2ranks.py
