target_sources(HiPACE
  PRIVATE
    BeamParticleContainer.cpp
    BeamParticleContainerInit.cpp
    PlasmaParticleContainer.cpp
    PlasmaParticleContainerInit.cpp
    BinSort.cpp
    BoxSort.cpp
    MultiBeam.cpp
)

add_subdirectory(deposition)
add_subdirectory(pusher)
add_subdirectory(profiles)
