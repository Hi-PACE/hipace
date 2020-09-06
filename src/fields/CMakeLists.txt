target_sources(HiPACE
  PRIVATE
    Fields.cpp
)

add_subdirectory(fft_poisson_solver)
add_subdirectory(filters)
