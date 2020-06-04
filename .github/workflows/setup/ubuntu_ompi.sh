#!/usr/bin/env bash
#
# Copyright 2020 The HiPACE Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential  \
    g++              \
    libfftw3-dev     \
    libopenmpi-dev   \
    openmpi-bin
