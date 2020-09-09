target_sources(HiPACE
  PRIVATE
    FFTPoissonSolver.cpp
    FFTPoissonSolverPeriodic.cpp
)

add_subdirectory(fft)
