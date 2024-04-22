/* Copyright 2020
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl, MaxThevenet, Remi Lehe, WeiqunZhang
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef FFT_POISSON_SOLVER_PERIODIC_H_
#define FFT_POISSON_SOLVER_PERIODIC_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include "FFTPoissonSolver.H"

#include <AMReX_MultiFab.H>
#include <AMReX_GpuComplex.H>

/** Declare type for fields in spectral fields */
using SpectralField = amrex::FabArray< amrex::BaseFab <amrex::GpuComplex<amrex::Real>> >;

/**
 * \brief This class handles functions and data to perform transverse Fourier-based Poisson solves.
 *
 * For a given source S, it solves equation Laplacian(F) = S and returns F.
 * Once an instance is created, a typical use consists in:
 * 1. Compute S directly in FFTPoissonSolver::m_stagingArea
 * 2. Call FFTPoissonSolver::SolvePoissonEquation(mf), which will solve Poisson equation with RHS
 *    in the staging area and return the LHS in mf.
 */
class FFTPoissonSolverPeriodic final : public FFTPoissonSolver
{
public:
    /** Constructor */
    FFTPoissonSolverPeriodic ( amrex::BoxArray const& realspace_ba,
                               amrex::DistributionMapping const& dm,
                               amrex::Geometry const& gm);

    /** virtual destructor */
    virtual ~FFTPoissonSolverPeriodic () override final {}

    /**
     * \brief Define real space and spectral space boxes and multifabs, multiplier
     * coefficients inv_k2 to solve Poisson equation and FFT plans.
     * Currently only works with a single box, i.e., serial FFT.
     *
     * \param[in] realspace_ba BoxArray on which the FFT is executed.
     * \param[in] dm DistributionMapping for the BoxArray.
     * \param[in] gm Geometry, contains the box dimensions.
     */
    void define ( amrex::BoxArray const& realspace_ba,
                  amrex::DistributionMapping const& dm,
                  amrex::Geometry const& gm);

    /**
     * Solve Poisson equation. The source term must be stored in the staging area m_stagingArea prior to this call.
     *
     * \param[in] lhs_mf Destination array, where the result is stored.
     */
    virtual void SolvePoissonEquation (amrex::MultiFab& lhs_mf) override final;

    /** Position and relative factor used to apply inhomogeneous Dirichlet boundary conditions
     * Note: inhomogeneous Dirichlet boundary conditions do not work with this solver
     */
    virtual amrex::Real BoundaryOffset() override final { return 1.; }
    virtual amrex::Real BoundaryFactor() override final { return 1.; }

private:
    /** Spectral fields, contains (complex) field in Fourier space */
    SpectralField m_tmpSpectralField;
    /** Multifab containing 1/(kx^2 + ky^2), to solve Poisson equation. */
    amrex::MultiFab m_inv_k2;
    /** forward FFT plan */
    AnyFFT m_forward_fft;
    /** backward FFT plan */
    AnyFFT m_backward_fft;
    /** work area for both FFT plans */
    amrex::Gpu::DeviceVector<char> m_fft_work_area;
};

#endif
