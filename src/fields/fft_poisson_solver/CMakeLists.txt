target_sources(Hipace
  PRIVATE
    FFTPoissonSolver.cpp
    FFTPoissonSolverPeriodic.cpp
    FFTPoissonSolverDirichlet.cpp
)

add_subdirectory(fft)
