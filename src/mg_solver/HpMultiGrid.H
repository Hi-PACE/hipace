/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_MULTIGRID_H_
#define HIPACE_MULTIGRID_H_

#include "utils/HipaceProfilerWrapper.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <type_traits>

/** brief namespace for Hipace Multigrid */
namespace hpmg {

/** \brief Multigrid solver
 *
 * This solves `-acoef * sol + Lap(sol) = rhs` with homogeneous Dirichlet BC
 * on a 2D slice.  It can solve two types of linear systems.
 *
 * (1) sol and rhs have two components, whereas acoef has only one
 *     component.  For Type I, call solve1(...).
 *
 * (2) acoef, sol, rhs are complex numbers. The system is equivalent to
 *       -acoef_real * sol_real + acoef_imag * sol_imag + Lap(sol_real) = rhs_real
 *       -acoef_imag * sol_real - acoef_real * sol_imag + Lap(sol_imag) = rhs_imag
 *     For Type II, call solve2(...).  Here, acoef_real and acoef_imag can be
 *     either a scalar constant or FArrayBox.
 */
class MultiGrid
{
public:

    /** \brief Ctor
     *
     * \param[in] geom Geometry describing a 2D slice
     */
    explicit MultiGrid (amrex::Geometry const& geom);

    /** \brief Dtor */
    ~MultiGrid ();

    /** \brief Solve the Type I equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve1 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs, amrex::FArrayBox const& acoef,
                 amrex::Real const tol_rel, amrex::Real const tol_abs, int const nummaxiter,
                 int const verbose);

    /** \brief Solve the Type II equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef_real the real part of the coefficient
     * \param[in] acoef_imag the imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    template <typename TR, typename TI>
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                 TR const& acoef_real, TI const& acoef_imag,
                 amrex::Real const tol_rel, amrex::Real const tol_abs,
                 int const nummaxiter, int const verbose);

    /** \brief Average down the coefficient.  Ideally, this function is not
     * supposed to be a public function.  It's made public due to a CUDA
     * limitation. */
    void average_down_acoef ();
    /** \brief Perform a V-cycle.  Ideally, this function is not supposed to
     * be a public function.  It's made public due to a CUDA limitation. */
    void vcycle ();
    /** \brief Solve at the bottom of the V-cycle.  Ideally, this function
     * is not supposed to be a public function.  It's made public due to a
     * CUDA limitation. */
    void bottomsolve ();
    /** \brief Private function used by solve1 and solve2.  It's made public
     * due to a CUDA limitation. */
    void solve_doit (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                     amrex::Real const tol_rel, amrex::Real const tol_abs,
                     int const nummaxiter, int const verbose);

private:

    static constexpr int m_num_system_types = 2;
    int m_system_type = 0;

    /** 2D slice domain */
    amrex::Vector<amrex::Box> m_domain;
    /** Cell sizes */
    amrex::Real m_dx, m_dy;

    /** Bottom MG level */
    int m_max_level;
    /** The level below which a single block kernel is used */
    int m_single_block_level_begin;
    /** Number of MG levels */
    int m_num_mg_levels;
    /** Number of single-block-kernel levels */
    int m_num_single_block_levels;

    /** Alias to the solution argument passed in solve() */
    amrex::FArrayBox m_sol;
    /** Alias to the RHS argument passed in solve() */
    amrex::FArrayBox m_rhs;

    /** Number of temporary fabs needed */
    static constexpr int nfabvs = 4;
    /** Fabs for coefficient, one for each level */
    amrex::Vector<amrex::FArrayBox> m_acf;
    /** Fabs for residual, one for each level */
    amrex::Vector<amrex::FArrayBox> m_res;
    /** Fabs for correction, one for each level */
    amrex::Vector<amrex::FArrayBox> m_cor;
    /** Fabs for residual of the residual-correction form, one for each level */
    amrex::Vector<amrex::FArrayBox> m_rescor;

    /** Device pointer to Array4s used by the single-block kernel at the bottom */
    amrex::Array4<amrex::Real> const* m_acf_a = nullptr;
    amrex::Array4<amrex::Real> const* m_res_a = nullptr;
    amrex::Array4<amrex::Real> const* m_cor_a = nullptr;
    amrex::Array4<amrex::Real> const* m_rescor_a = nullptr;

    /** Pinned vector as a staging area for memcpy to device */
    amrex::Gpu::PinnedVector<amrex::Array4<amrex::Real> > m_h_array4;
    /** Device vector of Array4s used by the single-block kernel at the bottom */
    amrex::Gpu::DeviceVector<amrex::Array4<amrex::Real> > m_d_array4;

#if defined(AMREX_USE_CUDA)
    /** CUDA graphs for average-down */
    bool m_cuda_graph_acf_created[m_num_system_types] = {false,false};
    cudaGraph_t m_cuda_graph_acf[m_num_system_types] = {NULL,NULL};
    cudaGraphExec_t m_cuda_graph_exe_acf[m_num_system_types] = {NULL,NULL};

    /** CUDA graphs for the V-cycle*/
    bool m_cuda_graph_vcycle_created[m_num_system_types] = {false,false};
    cudaGraph_t m_cuda_graph_vcycle[m_num_system_types] = {NULL,NULL};
    cudaGraphExec_t m_cuda_graph_exe_vcycle[m_num_system_types] = {NULL,NULL};
#endif
};

#if defined(AMREX_USE_GPU) || !defined(AMREX_USE_OMP)

using amrex::ParallelFor;

#else

// amrex::ParallelFor does not do OpenMP.  Thus we have hpmg::ParallelFor.

template <typename T, typename F>
void ParallelFor (T n, F&& f) noexcept
{
#pragma omp parallel for simd
    for (T i = 0; i < n; ++i) {
        f(i);
    }
}

template <typename F>
void ParallelFor (amrex::Box const& box, F&& f) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
#pragma omp parallel for collapse(2)
    for (int k = lo.z; k <= hi.z; ++k) {
    for (int j = lo.y; j <= hi.y; ++j) {
#pragma omp simd
    for (int i = lo.x; i <= hi.x; ++i) {
        f(i,j,k);
    }}}
}

template <typename F>
void ParallelFor (amrex::Box const& box, int ncomp, F&& f) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
#pragma omp parallel for collapse(3)
    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
#pragma omp simd
        for (int i = lo.x; i <= hi.x; ++i) {
            f(i,j,k,n);
        }}}
    }
}

#endif

template <typename TR, typename TI>
void MultiGrid::solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                        TR const& acoef_real, TI const& acoef_imag,
                        amrex::Real const tol_rel, amrex::Real const tol_abs,
                        int const nummaxiter, int const verbose)
{
    HIPACE_PROFILE("hpmg::MultiGrid::solve2()");
    m_system_type = 2;

    auto const& array_m_acf = m_acf[0].array();
    if constexpr (std::is_arithmetic<TR>::value && std::is_arithmetic<TI>::value)
    {
        hpmg::ParallelFor(m_acf[0].box(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                array_m_acf(i,j,0,0) = acoef_real;
                array_m_acf(i,j,0,1) = acoef_imag;
            });
    }
    else if constexpr (std::is_arithmetic<TR>::value)
    {
        AMREX_ALWAYS_ASSERT(amrex::makeSlab(acoef_imag.box(),2,0).contains(m_domain.front()));
        amrex::FArrayBox ifab(amrex::makeSlab(acoef_imag.box(), 2, 0),
                              1, acoef_imag.dataPtr());
        auto const& ai = ifab.const_array();
        hpmg::ParallelFor(m_acf[0].box(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                array_m_acf(i,j,0,0) = acoef_real;
                array_m_acf(i,j,0,1) = ai(i,j,0);
            });
    }
    else if constexpr (std::is_arithmetic<TI>::value)
    {
        AMREX_ALWAYS_ASSERT(amrex::makeSlab(acoef_real.box(),2,0).contains(m_domain.front()));
        amrex::FArrayBox rfab(amrex::makeSlab(acoef_real.box(), 2, 0),
                              1, acoef_real.dataPtr());
        auto const& ar = rfab.const_array();
        hpmg::ParallelFor(m_acf[0].box(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                array_m_acf(i,j,0,0) = ar(i,j,0);
                array_m_acf(i,j,0,1) = acoef_imag;
            });
    }
    else
    {
        AMREX_ALWAYS_ASSERT(amrex::makeSlab(acoef_real.box(),2,0).contains(m_domain.front()) &&
                            amrex::makeSlab(acoef_imag.box(),2,0).contains(m_domain.front()));
        amrex::FArrayBox rfab(amrex::makeSlab(acoef_real.box(), 2, 0),
                              1, acoef_real.dataPtr());
        amrex::FArrayBox ifab(amrex::makeSlab(acoef_imag.box(), 2, 0),
                              1, acoef_imag.dataPtr());
        auto const& ar = rfab.const_array();
        auto const& ai = ifab.const_array();
        hpmg::ParallelFor(m_acf[0].box(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                array_m_acf(i,j,0,0) = ar(i,j,0);
                array_m_acf(i,j,0,1) = ai(i,j,0);
            });

    }

    average_down_acoef();

    solve_doit(sol, rhs, tol_rel, tol_abs, nummaxiter, verbose);
}

}

#endif
