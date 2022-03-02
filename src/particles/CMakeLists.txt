# Copyright 2020-2021 Andrew Myers, Axel Huebl, MaxThevenet
# Remi Lehe, Severin Diederichs, atmyers
#
#
# This file is part of HiPACE++.
#
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
