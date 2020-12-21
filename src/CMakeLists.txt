target_sources(HiPACE
  PRIVATE
    main.cpp
    Hipace.cpp
)

add_subdirectory(fields)
add_subdirectory(particles)
add_subdirectory(utils)
add_subdirectory(diagnostics)
