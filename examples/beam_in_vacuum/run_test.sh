#! /usr/bin/env sh

# This file is part of the Hipace test suite.
# It runs a Hipace simulation for a can beam in vacuum, and compares the result
# of the simulation to theory.

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_EXAMPLE_DIR=$2

# Run the simulation
$HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs hipace.depos_order_xy=1

# Analyse the results
python $HIPACE_EXAMPLE_DIR/analysis.py
