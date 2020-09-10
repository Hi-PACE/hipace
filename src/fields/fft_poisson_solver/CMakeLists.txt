target_sources(HiPACE
  PRIVATE
    FFTPoissonSolverUtil.cpp
    FFTPoissonSolver.cpp
    FFTPoissonSolverPeriodic.cpp
    FFTPoissonSolverDirichlet.cpp
)

add_subdirectory(fft)
