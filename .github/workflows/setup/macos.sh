#!/usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

set +e
rm -rf /usr/local/bin/2to3
brew update
brew install ccache
brew install cmake
brew install open-mpi
set -e
