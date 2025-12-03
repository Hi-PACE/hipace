#! /usr/bin/env bash

# Copyright 2025
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
#
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

echo $HIPACE_EXECUTABLE

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/blowout_wake
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

RTOL=2e-3


for solver_type in FFTDirichletDirect FFTDirichletExpanded FFTDirichletFast FFTDirichletQuick MGDirichlet
do

    echo "Testing $solver_type"

    # Run the simulation
    mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI \
            hipace.output_folder = $solver_type/ \
            fields.poisson_solver = $solver_type \
            amr.n_cell = 64 72 100 \
            max_step = 0 \
            MGDirichlet.MG_tolerance_rel = 1e-7 \

    # Compare the results with checksum benchmark
    $HIPACE_TEST_DIR/checksum/checksumAPI.py \
        --evaluate \
        --rtol $RTOL \
        --file_name $solver_type/hdf5/ \
        --test-name Poisson_even.1Rank

    rm -rf $solver_type

done
