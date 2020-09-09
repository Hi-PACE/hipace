target_sources(HiPACE
  PRIVATE
    FFTPoissonSolver.cpp
    FFTPoissonSolverPeriodic.cpp
    FFTPoissonSolverDirichlet.cpp
)

add_subdirectory(fft)
