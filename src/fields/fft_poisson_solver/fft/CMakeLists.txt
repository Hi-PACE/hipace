# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet, Remi Lehe
# License: BSD-3-Clause-LBNL

if (HiPACE_COMPUTE STREQUAL CUDA)
  target_sources(HiPACE
    PRIVATE
        WrapCuFFT.cpp
        WrapCuDST.cpp
        CuFFTUtils.cpp
  )
elseif(HiPACE_COMPUTE STREQUAL HIP)
  target_sources(HiPACE
    PRIVATE
        WrapRocFFT.cpp
        WrapRocDST.cpp
        RocFFTUtils.cpp
  )
else()
  target_sources(HiPACE
    PRIVATE
        WrapFFTW.cpp
        WrapDSTW.cpp
  )
endif()
