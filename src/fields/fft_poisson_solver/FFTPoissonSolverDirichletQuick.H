/* Copyright 2025
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef FFT_POISSON_SOLVER_DIRICHLET_QUICK_H_
#define FFT_POISSON_SOLVER_DIRICHLET_QUICK_H_

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
class FFTPoissonSolverDirichletQuick final : public FFTPoissonSolver
{
public:
    /** Constructor */
    FFTPoissonSolverDirichletQuick ( amrex::BoxArray const& a_realspace_ba,
                                    amrex::DistributionMapping const& dm,
                                    amrex::Geometry const& gm);

    /** virtual destructor */
    virtual ~FFTPoissonSolverDirichletQuick () override final {}

    /**
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

    /** Position and relative factor used to apply inhomogeneous Dirichlet boundary conditions */
    virtual amrex::Real BoundaryOffset() override final { return 0.5; }
    virtual amrex::Real BoundaryFactor() override final { return 2.; }

private:
    /** Real array for the FFTs */
    amrex::Gpu::DeviceVector<amrex::Real> m_real_array;
    /** Complex array for the FFTs */
    amrex::Gpu::DeviceVector<amrex::GpuComplex<amrex::Real>> m_comp_array;
    /** FFT plan in x direction */
    AnyFFT m_x_r2cfft;
    /** FFT plan in y direction */
    AnyFFT m_y_r2cfft;
    /** FFT plan in x direction */
    AnyFFT m_x_c2rfft;
    /** FFT plan in y direction */
    AnyFFT m_y_c2rfft;
    /** work area for all DST plans */
    amrex::Gpu::DeviceVector<char> m_fft_work_area;
    /** x prefactor for dst2_out and dst3_in */
    amrex::Gpu::DeviceVector<amrex::GpuComplex<amrex::Real>> m_omega_x;
    /** y prefactor for dst2_out and dst3_in */
    amrex::Gpu::DeviceVector<amrex::GpuComplex<amrex::Real>> m_omega_y;
    /** x prefactor for eigenvalue */
    amrex::Gpu::DeviceVector<amrex::Real> m_eig_x;
    /** y prefactor for eigenvalue */
    amrex::Gpu::DeviceVector<amrex::Real> m_eig_y;
};

#endif
