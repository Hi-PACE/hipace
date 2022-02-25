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
    CoulombCollision.cpp
)

add_subdirectory(deposition)
add_subdirectory(pusher)
add_subdirectory(profiles)
