# Copyright 2020-2021
#
# This file is part of HiPACE++.
#
# Authors: Andrew Myers, Axel Huebl, MaxThevenet, Remi Lehe
# Severin Diederichs, atmyers
# License: BSD-3-Clause-LBNL

target_sources(HiPACE
  PRIVATE
    BeamParticleContainer.cpp
    BeamParticleContainerInit.cpp
    PlasmaParticleContainer.cpp
    PlasmaParticleContainerInit.cpp
    SliceSort.cpp
    TileSort.cpp
    BoxSort.cpp
    MultiBeam.cpp
    MultiPlasma.cpp
)

add_subdirectory(deposition)
add_subdirectory(pusher)
add_subdirectory(profiles)
