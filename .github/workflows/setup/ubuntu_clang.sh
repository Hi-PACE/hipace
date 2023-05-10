#!/usr/bin/env bash

# Copyright 2020-2022
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential     \
    ccache              \
    clang-7             \
    libopenmpi-dev      \
    openmpi-bin
