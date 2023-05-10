#!/usr/bin/env bash

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

#
# Copyright 2021-2022 Axel Huebl
#
# License: BSD-3-Clause-LBNL

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    environment-modules \
    gnupg               \
    libhiredis-dev      \
    libzstd-dev         \
    ninja-build         \
    pkg-config          \
    wget

wget -q https://developer.download.nvidia.com/hpc-sdk/21.9/nvhpc-21-9_21.9_amd64.deb \
        https://developer.download.nvidia.com/hpc-sdk/21.9/nvhpc-2021_21.9_amd64.deb
sudo apt-get update
sudo apt-get install -y ./nvhpc-21-9_21.9_amd64.deb ./nvhpc-2021_21.9_amd64.deb
rm -rf ./nvhpc-21-9_21.9_amd64.deb ./nvhpc-2021_21.9_amd64.deb

# things should reside in /opt/nvidia/hpc_sdk now

# activation via:
#   source /etc/profile.d/modules.sh
#   module load /opt/nvidia/hpc_sdk/modulefiles/nvhpc/21.9

# cmake-easyinstall
#
sudo curl -L -o /usr/local/bin/cmake-easyinstall https://raw.githubusercontent.com/ax3l/cmake-easyinstall/main/cmake-easyinstall
sudo chmod a+x /usr/local/bin/cmake-easyinstall
export CEI_SUDO="sudo"
export CEI_TMP="/tmp/cei"

# ccache 4.2+
#
CXXFLAGS="" cmake-easyinstall --prefix=/usr/local \
    git+https://github.com/ccache/ccache.git@v4.6 \
    -DCMAKE_BUILD_TYPE=Release        \
    -DENABLE_DOCUMENTATION=OFF        \
    -DENABLE_TESTING=OFF              \
    -DWARNINGS_AS_ERRORS=OFF
