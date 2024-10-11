/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_MULTIGRID_H_
#define HIPACE_MULTIGRID_H_

#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <type_traits>

/** brief namespace for Hipace Multigrid */
namespace hpmg {

/** \brief Multigrid solver
 *
 * This solves `-acoef * sol + Lap(sol) = rhs` with homogeneous Dirichlet BC
 * on a 2D slice.  It can solve three types of linear systems.
 *
 * (1) sol and rhs have two components, whereas acoef has only one
 *     component.  For Type I, call solve1(...).
 *
 * (2) acoef, sol, rhs are complex numbers. The system is equivalent to
 *       -acoef_real * sol_real + acoef_imag * sol_imag + Lap(sol_real) = rhs_real
 *       -acoef_imag * sol_real - acoef_real * sol_imag + Lap(sol_imag) = rhs_imag
 *     For Type II, call solve2(...).  Here, acoef_real and acoef_imag can be
 *     either a scalar constant or FArrayBox.
 *
 * (3) sol and rhs have one component, whereas acoef is zero everywhere.
 *     For Type III, call solve3(...).
 */
class MultiGrid
{
public:

    /** \brief Ctor
     *
     * \param[in] dx Cell spacing in x direction
     * \param[in] dy Cell spacing in y direction
     * \param[in] a_domain Box describing a 2D slice
     * \param[in] a_system_type which system type should be solved
     */
    explicit MultiGrid (amrex::Real dx, amrex::Real dy, amrex::Box a_domain, int a_system_type);

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
     * \param[in] acoef_real the constant real part of the coefficient
     * \param[in] acoef_imag the constant imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                 amrex::Real const acoef_real, amrex::Real const acoef_imag,
                 amrex::Real const tol_rel, amrex::Real const tol_abs,
                 int const nummaxiter, int const verbose);

    /** \brief Solve the Type II equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef_real the constant real part of the coefficient
     * \param[in] acoef_imag the array imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                amrex::Real const acoef_real, amrex::FArrayBox const& acoef_imag,
                amrex::Real const tol_rel, amrex::Real const tol_abs,
                int const nummaxiter, int const verbose);

    /** \brief Solve the Type II equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef_real the array real part of the coefficient
     * \param[in] acoef_imag the constant imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                amrex::FArrayBox const& acoef_real, amrex::Real const acoef_imag,
                amrex::Real const tol_rel, amrex::Real const tol_abs,
                int const nummaxiter, int const verbose);

    /** \brief Solve the Type II equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef_real the array real part of the coefficient
     * \param[in] acoef_imag the array imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                amrex::FArrayBox const& acoef_real, amrex::FArrayBox const& acoef_imag,
                amrex::Real const tol_rel, amrex::Real const tol_abs,
                int const nummaxiter, int const verbose);

    /** \brief Solve the Type III equation given the initial guess and right hand side.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve3 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                 amrex::Real const tol_rel, amrex::Real const tol_abs, int const nummaxiter,
                 int const verbose);

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
    /** \brief Centers the input box in x and y around the domain so that only the ghost
     * cells "overhang". Make it a slab in the z direction and set the index to 0.
     */
    static amrex::Box center_box (amrex::Box in_box, amrex::Box domain) {
        amrex::Box out_box = amrex::makeSlab(
            in_box + (domain.smallEnd() + domain.bigEnd() - in_box.smallEnd() - in_box.bigEnd())/2,
        2, 0);
        out_box.setType(domain.ixType());
        AMREX_ALWAYS_ASSERT(out_box.contains(domain));
        return out_box;
    }

    /** \brief Return the number of components used for a given system type
     *
     * \param[in] system_type which system type to use
     */
    static constexpr int get_num_comps (int system_type) {
        switch (system_type) {
            case 1: return 2;
            case 2: return 2;
            case 3: return 1;
        }
        return 0;
    }

    /** \brief Return the number of acf components used for a given system type
     *
     * \param[in] system_type which system type to use
     */
    static constexpr int get_num_comps_acf (int system_type) {
        switch (system_type) {
            case 1: return 1;
            case 2: return 2;
            case 3: return 0;
        }
        return 0;
    }

    /** When applying Dirichlet boundary conditions, shift boundary value by offset number of cells */
    amrex::Real m_boundary_condition_offset = 0.;
    /** When applying Dirichlet boundary conditions, multiply the boundary value by this factor */
    amrex::Real m_boundary_condition_factor = 0.;

private:

    static constexpr int m_num_system_types = 3;
    int m_system_type = 0;
    int m_num_comps = 0;
    int m_num_comps_acf = 0;

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
    /** If the single block kernel should be used */
    bool m_use_single_block_kernel = true;

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
    bool m_cuda_graph_acf_created = false;
    cudaGraph_t m_cuda_graph_acf = NULL;
    cudaGraphExec_t m_cuda_graph_exe_acf = NULL;

    /** CUDA graphs for the V-cycle*/
    std::map<std::pair<const amrex::Real*, const amrex::Real*>,
             std::pair<cudaGraph_t, cudaGraphExec_t>> m_cuda_graph_vcycle;
#endif
};

#if defined(AMREX_USE_GPU) || !defined(AMREX_USE_OMP)

using amrex::ParallelFor;

#else

// amrex::ParallelFor does not do OpenMP.  Thus we have hpmg::ParallelFor.

template <typename T, typename F>
void ParallelFor (T n, F&& f) noexcept
{
    if (n < 1000) {
        for (T i = 0; i < n; ++i) {
            f(i);
        }
    } else {
#pragma omp parallel for simd
        for (T i = 0; i < n; ++i) {
            f(i);
        }
    }
}

template <typename F>
void ParallelFor (amrex::BoxND<2> const& box, F&& f) noexcept
{
    if (box.numPts() < 1000) {
        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);
        for (int j = lo.y; j <= hi.y; ++j) {
#pragma omp simd
        for (int i = lo.x; i <= hi.x; ++i) {
            f(i,j);
        }}
    } else {
        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);

        constexpr int tilesize = 8;
#pragma omp parallel for collapse(2)
        for (int jt = lo.y; jt <= hi.y; jt+=tilesize) {
        for (int it = lo.x; it <= hi.x; it+=tilesize) {
        for (int j = jt; j < jt+tilesize && j<=hi.y; ++j) {
        for (int i = it; i < it+tilesize && i<=hi.x; ++i) {
            f(i,j);
        }}}}
    }
}

template <typename F>
void ParallelFor (amrex::BoxND<2> const& box, int ncomp, F&& f) noexcept
{
    if (box.numPts()*ncomp < 1000) {
        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);
        for (int n = 0; n < ncomp; ++n) {
            for (int j = lo.y; j <= hi.y; ++j) {
#pragma omp simd
            for (int i = lo.x; i <= hi.x; ++i) {
                f(i,j,n);
            }}
        }
    } else {
        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);

        constexpr int tilesize = 8;
#pragma omp parallel for collapse(3)
        for (int n = 0; n < ncomp; ++n) {
            for (int jt = lo.y; jt <= hi.y; jt+=tilesize) {
            for (int it = lo.x; it <= hi.x; it+=tilesize) {
            for (int j = jt; j < jt+tilesize && j<=hi.y; ++j) {
            for (int i = it; i < it+tilesize && i<=hi.x; ++i) {
                f(i,j,n);
            }}}}
        }
    }
}

#endif

}

#endif
