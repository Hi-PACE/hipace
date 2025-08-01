#! /usr/bin/env bash
#
# This file is part of HiPACE++.
#
# Authors: Xingjian Hui
# This file is part of the HiPACE++ test suite.
# It tests the STC initialisation with LASY functions.

# abort on first encountered error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2
HIPACE_COMPUTE=$3

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/laser_STC
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

rm -rf $TEST_NAME

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_STC \
        hipace.file_prefix = $TEST_NAME
# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_STC.py --output-dir=$TEST_NAME
