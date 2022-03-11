#!/usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential     \
    g++-8               \
    libopenmpi-dev      \
    openmpi-bin         \
    nvidia-cuda-dev     \
    nvidia-cuda-toolkit
