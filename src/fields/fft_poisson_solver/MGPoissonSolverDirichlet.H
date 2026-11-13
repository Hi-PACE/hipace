/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef MG_POISSON_SOLVER_DIRICHLET_H_
#define MG_POISSON_SOLVER_DIRICHLET_H_

#include "FFTPoissonSolver.H"
#include "mg_solver/HpMultiGrid.H"

#include <AMReX_MultiFab.H>

/**
 * \brief This class handles functions and data to perform transverse multigrid-based Poisson solves.
 *
 * For a given source S, it solves equation Laplacian(F) = S and returns F.
 * Once an instance is created, a typical use consists in:
 * 1. Compute S directly in FFTPoissonSolver::m_stagingArea
 * 2. Call FFTPoissonSolver::SolvePoissonEquation(mf), which will solve Poisson equation with RHS
 *    in the staging area and return the LHS in mf.
 */
class MGPoissonSolverDirichlet final : public FFTPoissonSolver
{
public:
    /** Constructor */
    MGPoissonSolverDirichlet ( amrex::BoxArray const& a_realspace_ba,
                               amrex::DistributionMapping const& dm,
                               amrex::Geometry const& gm);

    /** virtual destructor */
    ~MGPoissonSolverDirichlet () final = default;

    MGPoissonSolverDirichlet (const MGPoissonSolverDirichlet& rhs) = delete;
    MGPoissonSolverDirichlet (MGPoissonSolverDirichlet&& rhs) = delete;
    MGPoissonSolverDirichlet& operator= (const MGPoissonSolverDirichlet& rhs) = delete;
    MGPoissonSolverDirichlet& operator= (MGPoissonSolverDirichlet&& rhs) = delete;

    /**
     * Solve Poisson equation. The source term must be stored in the staging area m_stagingArea prior to this call.
     *
     * \param[in] lhs_mf Destination array, where the result is stored.
     */
    void SolvePoissonEquation (amrex::MultiFab& lhs_mf) final;

    /** Position and relative factor used to apply inhomogeneous Dirichlet boundary conditions */
    amrex::Real BoundaryOffset() final { return m_mg->m_boundary_condition_offset; }
    amrex::Real BoundaryFactor() final { return m_mg->m_boundary_condition_factor; }

private:
    amrex::Real m_MG_tolerance_rel = 1.e-4;
    amrex::Real m_MG_tolerance_abs = 0.;
    int m_MG_verbose = 0;
    /** hpmg solver for the poisson solver */
    std::unique_ptr<hpmg::MultiGrid> m_mg;
};

#endif
