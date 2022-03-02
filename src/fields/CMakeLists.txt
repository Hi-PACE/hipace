# Copyright 2020 Andrew Myers, MaxThevenet, Remi Lehe
#
#
# This file is part of HiPACE++.
#
# License: BSD-3-Clause-LBNL

target_sources(HiPACE
  PRIVATE
    Fields.cpp
)

add_subdirectory(fft_poisson_solver)
