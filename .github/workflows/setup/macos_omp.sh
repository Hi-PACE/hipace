#!/usr/bin/env bash
#
# Copyright 2020 The Hipace Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

brew update
brew install cmake
brew install libomp
brew install open-mpi
