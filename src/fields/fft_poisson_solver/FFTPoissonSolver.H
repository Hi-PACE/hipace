/* Copyright 2020
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl, MaxThevenet, Remi Lehe, WeiqunZhang
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef FFT_POISSON_SOLVER_H_
#define FFT_POISSON_SOLVER_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"

#include <AMReX_MultiFab.H>
#include <AMReX_GpuComplex.H>

/**
 * \brief This class handles functions and data to perform transverse Fourier-based Poisson solves.
 *
 * For a given source S, it solves equation Laplacian(F) = S and returns F.
 * Once an instance is created, a typical use consists in:
 * 1. Compute S directly in FFTPoissonSolver::m_stagingArea
 * 2. Call FFTPoissonSolver::SolvePoissonEquation(mf), which will solve Poisson equation with RHS
 *    in the staging area and return the LHS in mf.
 */
class FFTPoissonSolver
{
public:

    /** Default constructor */
    FFTPoissonSolver () = default;

    /** Abstract class needs a virtual destructor */
    virtual ~FFTPoissonSolver () = 0;

    /**
     * Solve Poisson equation. The source term must be stored in the staging area m_stagingArea prior to this call.
     *
     * \param[in] lhs_mf Destination array, where the result is stored.
     */
    virtual void SolvePoissonEquation (amrex::MultiFab& lhs_mf) = 0;

    /** Position and relative factor used to apply inhomogeneous Dirichlet boundary conditions */
    virtual amrex::Real BoundaryOffset() = 0;
    virtual amrex::Real BoundaryFactor() = 0;

    /** Get reference to the taging area */
    amrex::MultiFab& StagingArea ();
protected:
    /** Staging area, contains (real) field in real space.
     * This is where the source term is stored before calling the Poisson solver
     *
     * The staging area does not cover ghost cells for all boundary condition types,
     * however the MultFab has one ghost cell when using the MG poisson solver.
     */
    amrex::MultiFab m_stagingArea;
};

#endif
