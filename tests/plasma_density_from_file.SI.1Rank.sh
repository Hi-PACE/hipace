#! /usr/bin/env bash

# Copyright 2025
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, MaxThevenet
# License: BSD-3-Clause-LBNL

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/plasma_initialization
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

# gererate plasma profiles
python3 ${HIPACE_SOURCE_DIR}/tools/write_plasma_density.py
python3 ${HIPACE_SOURCE_DIR}/tools/write_plasma_density_rz.py

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_SI_plasma_from_file \
        hipace.file_prefix=${TEST_NAME} \

$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
