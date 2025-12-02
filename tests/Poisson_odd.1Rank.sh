#! /usr/bin/env bash

# Copyright 2025
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
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

echo $HIPACE_EXECUTABLE

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# Relative tolerance for checksum tests depends on the platform
RTOL=1e-12 && [[ "$HIPACE_EXECUTABLE" == *"hipace"*".CUDA."* ]] && RTOL=2e-5


for solver_type in FFTDirichletDirect FFTDirichletExpanded FFTDirichletFast FFTDirichletQuick MGDirichlet
do

echo "Testing $solver_type"

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
        hipace.output_folder=$solver_type/ \
        fields.poisson_solver = $solver_type \
        amr.n_cell = 63 71 100 \
        max_step=0

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --rtol $RTOL \
    --file_name $solver_type/hdf5/ \
    --test-name Poisson_odd.1Rank

rm -rf $solver_type

done
