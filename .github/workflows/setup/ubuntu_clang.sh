#!/usr/bin/env bash
#
# Copyright 2020 The HiPACE++ Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential     \
    clang-4.0           \
    libopenmpi-dev      \
    openmpi-bin
