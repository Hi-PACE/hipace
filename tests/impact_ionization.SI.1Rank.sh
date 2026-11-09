#! /usr/bin/env bash

# Copyright 2026
#
# This file is part of HiPACE++.
#
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It runs a simulation where thermal electrons ionize neutral hydrogen by impact,
# compares the ionized fraction with theory, and compares with the checksum benchmark

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/impact_ionization
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

rm -rf $TEST_NAME

# Run the simulation
OMP_NUM_THREADS=1 mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_impact_ionization_SI \
        hipace.file_prefix=$TEST_NAME \
        plasmas.insitu_file_prefix=$TEST_NAME/insitu

# Compare the ionized fraction with theory
python3 $HIPACE_EXAMPLE_DIR/analysis_impact_ionization.py --insitu=$TEST_NAME/insitu

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME