# Copyright 2020
#
# This file is part of HiPACE++.
#
# Authors: Andrew Myers, MaxThevenet, Remi Lehe
# License: BSD-3-Clause-LBNL

target_sources(HiPACE
  PRIVATE
    Fields.cpp
)

add_subdirectory(fft_poisson_solver)
