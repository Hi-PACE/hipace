#!/usr/bin/env bash

# Copyright 2020-2022
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential     \
    ccache              \
    clang-7             \
    libopenmpi-dev      \
    openmpi-bin
