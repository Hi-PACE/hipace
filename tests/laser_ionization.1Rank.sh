#! /usr/bin/env bash

#
# This file is part of HiPACE++.
#
# Authors: EyaDammak


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation with in neutral hydrogen that gets ionized by the laser

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2
HIPACE_COMPUTE=$3

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/laser_ionization
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

rm -rf $TEST_NAME

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_laser_ionization \
    my_constants.a0 = 0.00885126 \
    plasmas.do_push = 0 \
    hipace.file_prefix=$TEST_NAME/linear \
    hipace.temperature_depos_order = 0 \
    plasmas.insitu_file_prefix = $TEST_NAME/insitu_linear

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_laser_ionization \
    my_constants.a0 = 0.00787934 \
    lasers.polarization = circular \
    plasmas.do_push = 0 \
    hipace.file_prefix=$TEST_NAME/circular \
    hipace.temperature_depos_order = 0 \
    plasmas.insitu_file_prefix = $TEST_NAME/insitu_circular

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_ionization.py --diags_linear=$TEST_NAME/linear  --diags_circular=$TEST_NAME/circular \
    --insitu_linear=$TEST_NAME/insitu_linear --insitu_circular=$TEST_NAME/insitu_circular


# Compare the results with checksum benchmark if it runs on CPU only
if [[ "$HIPACE_COMPUTE" != "CUDA" ]]; then
     $HIPACE_TEST_DIR/checksum/checksumAPI.py \
         --evaluate \
         --file_name $TEST_NAME/linear \
         --test-name $TEST_NAME
 fi
