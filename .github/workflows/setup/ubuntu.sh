#!/usr/bin/env bash
#
# Copyright 2020 The HiPACE Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential \
    libfftw3-dev    \
    g++ \
    python3 \
    python3-pip

python -m pip install --upgrade pip
python -m pip install --upgrade matplotlib==3.2.2 numpy scipy yt
