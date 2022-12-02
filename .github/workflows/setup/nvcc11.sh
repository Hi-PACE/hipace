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
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    g++                 \
    gnupg               \
    libhiredis-dev      \
    libopenmpi-dev      \
    libzstd-dev         \
    ninja-build         \
    openmpi-bin         \
    pkg-config          \
    wget

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" \
    | sudo tee /etc/apt/sources.list.d/cuda.list

sudo apt-get update
sudo apt-get install -y          \
    cuda-command-line-tools-11-8 \
    cuda-compiler-11-8           \
    cuda-cupti-dev-11-8          \
    cuda-minimal-build-11-8      \
    cuda-nvml-dev-11-8           \
    cuda-nvtx-11-8               \
    libcufft-dev-11-8            \
    libcurand-dev-11-8
sudo ln -s cuda-11.8 /usr/local/cuda

# cmake-easyinstall
#
sudo curl -L -o /usr/local/bin/cmake-easyinstall https://git.io/JvLxY
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
