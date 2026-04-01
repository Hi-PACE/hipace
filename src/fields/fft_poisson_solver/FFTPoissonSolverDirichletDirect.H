/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef FFT_POISSON_SOLVER_DIRICHLET_DIRECT_H_
#define FFT_POISSON_SOLVER_DIRICHLET_DIRECT_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include "FFTPoissonSolver.H"

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
class FFTPoissonSolverDirichletDirect final : public FFTPoissonSolver
{
public:
    /** Constructor */
    FFTPoissonSolverDirichletDirect ( amrex::BoxArray const& a_realspace_ba,
                                      amrex::DistributionMapping const& dm,
                                      amrex::Geometry const& gm,
                                      bool is_even);

    /** virtual destructor */
    virtual ~FFTPoissonSolverDirichletDirect () override final {}

    /**
     * \brief Define real space and spectral space boxes and multifabs, Dirichlet
     * eigenvalue matrix m_eigenvalue_matrix to solve Poisson equation and FFT plans.
     * Currently only works with a single box, i.e., serial FFT.
     *
     * \param[in] realspace_ba BoxArray on which the FFT is executed.
     * \param[in] dm DistributionMapping for the BoxArray.
     * \param[in] gm Geometry, contains the box dimensions.
     * \param[in] is_even True: use DST-2 / DST-3; False: use DST-1
     */
    void define ( amrex::BoxArray const& realspace_ba,
                  amrex::DistributionMapping const& dm,
                  amrex::Geometry const& gm,
                  bool is_even);

    /**
     * Solve Poisson equation. The source term must be stored in the staging area m_stagingArea prior to this call.
     *
     * \param[in] lhs_mf Destination array, where the result is stored.
     */
    virtual void SolvePoissonEquation (amrex::MultiFab& lhs_mf) override final;

    /** Position and relative factor used to apply inhomogeneous Dirichlet boundary conditions */
    virtual amrex::Real BoundaryOffset() override final { return m_is_even ? 0.5 : 1.; }
    virtual amrex::Real BoundaryFactor() override final { return m_is_even ? 2. : 1.; }

private:
    /** Spectral fields, contains (real) field in Fourier space */
    amrex::MultiFab m_tmpSpectralField;
    /** Multifab eigenvalues, to solve Poisson equation with Dirichlet BC. */
    amrex::MultiFab m_eigenvalue_matrix;
    /** forward DST plan */
    AnyFFT m_forward_fft;
    /** backward DST plan */
    AnyFFT m_backward_fft;
    /** work area for both DST plans */
    amrex::Gpu::DeviceVector<char> m_fft_work_area;
    /** True: use DST-2 / DST-3; False: use DST-1 */
    bool m_is_even;
};

#endif
