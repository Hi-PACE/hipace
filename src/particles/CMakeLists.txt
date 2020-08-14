target_sources(HiPACE
  PRIVATE
    BeamParticleContainer.cpp
    BeamParticleContainerInit.cpp
    PlasmaParticleContainer.cpp
    PlasmaParticleContainerInit.cpp
)

add_subdirectory(deposition)
add_subdirectory(pusher)
