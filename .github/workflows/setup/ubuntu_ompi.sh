#!/usr/bin/env bash

# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
#
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
    g++                 \
    libfftw3-dev        \
    libopenmpi-dev      \
    openmpi-bin         \
    libhdf5-openmpi-dev \
    python3             \
    python3-pip         \
    python3-setuptools

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 2
sudo update-alternatives --set python /usr/bin/python3

python -m pip install --upgrade pip
python -m pip install --upgrade matplotlib numpy scipy openpmd-viewer openpmd-api
