#!/usr/bin/env bash

# Copyright 2020-2021 Axel Huebl, MaxThevenet, Severin Diederichs
#
#
# This file is part of HiPACE++.
#
# License: BSD-3-Clause-LBNL

#
# Copyright 2020 The HiPACE++ Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

brew update
brew install cmake
brew install open-mpi
