# Copyright 2020
#
# This file is part of HiPACE++.
#
# Authors: Andrew Myers, Axel Huebl, MaxThevenet, Severin Diederichs
# atmyers
# License: BSD-3-Clause-LBNL

target_sources(HiPACE
  PRIVATE
    main.cpp
    Hipace.cpp
    HipaceVersion.cpp
)

add_subdirectory(fields)
add_subdirectory(helmholtz)
add_subdirectory(particles)
add_subdirectory(laser)
add_subdirectory(utils)
add_subdirectory(diagnostics)
add_subdirectory(mg_solver)
add_subdirectory(salame)
