# Copyright 2020 Andrew Myers, Axel Huebl, MaxThevenet
# Severin Diederichs, atmyers
#
# This file is part of HiPACE++.
#
# License: BSD-3-Clause-LBNL

target_sources(HiPACE
  PRIVATE
    main.cpp
    Hipace.cpp
)

add_subdirectory(fields)
add_subdirectory(particles)
add_subdirectory(utils)
add_subdirectory(diagnostics)
