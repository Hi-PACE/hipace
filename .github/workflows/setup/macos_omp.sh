#!/usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL

#
# Copyright 2020 The HiPACE++ Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

set +e
rm -rf /usr/local/bin/2to3
brew update
brew install cmake
brew install libomp
brew install open-mpi
set -e
