target_sources(HiPACE
  PRIVATE
    main.cpp
    Hipace.cpp
    Constants.cpp
    AdaptiveTimeStep.cpp
)

add_subdirectory(fields)
add_subdirectory(particles)
