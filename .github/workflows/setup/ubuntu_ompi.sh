#!/usr/bin/env bash
#
# Copyright 2020 The HiPACE Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential     \
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
python -m pip install --upgrade matplotlib==3.2.2 numpy scipy openpmd-viewer
