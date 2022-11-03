/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#include "HpMultiGrid.H"
#include "utils/GPUUtil.H"
#include <algorithm>

using namespace amrex;

namespace hpmg {

namespace {

constexpr int n_cell_single = 32; // switch to single block when box is smaller than this

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real laplacian (int i, int j, int n, int ilo, int jlo, int ihi, int jhi,
                Array4<Real> const& phi, Real facx, Real facy)
{
    Real lap = Real(-2.)*(facx+facy)*phi(i,j,0,n);
    if (i == ilo) {
        lap += facx * (Real(4./3.)*phi(i+1,j,0,n) - Real(2.)*phi(i,j,0,n));
    } else if (i == ihi) {
        lap += facx * (Real(4./3.)*phi(i-1,j,0,n) - Real(2.)*phi(i,j,0,n));
    } else {
        lap += facx * (phi(i-1,j,0,n) + phi(i+1,j,0,n));
    }
    if (j == jlo) {
        lap += facy * (Real(4./3.)*phi(i,j+1,0,n) - Real(2.)*phi(i,j,0,n));
    } else if (j == jhi) {
        lap += facy * (Real(4./3.)*phi(i,j-1,0,n) - Real(2.)*phi(i,j,0,n));
    } else {
        lap += facy * (phi(i,j-1,0,n) + phi(i,j+1,0,n));
    }
    return lap;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real residual1 (int i, int j, int n, int ilo, int jlo, int ihi, int jhi,
                Array4<Real> const& phi, Real rhs, Real acf, Real facx, Real facy)
{
    Real lap = laplacian(i,j,n,ilo,jlo,ihi,jhi,phi,facx,facy);
    return rhs + acf*phi(i,j,0,n) - lap;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real residual2r (int i, int j, int ilo, int jlo, int ihi, int jhi,
                 Array4<Real> const& phi, Real rhs, Real acf_r, Real acf_i,
                 Real facx, Real facy)
{
    Real lap = laplacian(i,j,0,ilo,jlo,ihi,jhi,phi,facx,facy);
    return rhs + acf_r*phi(i,j,0,0) - acf_i*phi(i,j,0,1) - lap;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real residual2i (int i, int j, int ilo, int jlo, int ihi, int jhi,
                 Array4<Real> const& phi, Real rhs, Real acf_r, Real acf_i,
                 Real facx, Real facy)
{
    Real lap = laplacian(i,j,1,ilo,jlo,ihi,jhi,phi,facx,facy);
    return rhs + acf_i*phi(i,j,0,0) + acf_r*phi(i,j,0,1) - lap;
}

// res = rhs - L(phi)
void compute_residual (Box const& box, Array4<Real> const& res,
                       Array4<Real> const& phi, Array4<Real const> const& rhs,
                       Array4<Real const> const& acf, Real dx, Real dy,
                       int system_type)
{
    int const ilo = box.smallEnd(0);
    int const jlo = box.smallEnd(1);
    int const ihi = box.bigEnd(0);
    int const jhi = box.bigEnd(1);
    Real facx = Real(1.)/(dx*dx);
    Real facy = Real(1.)/(dy*dy);
    if (system_type == 1) {
        hpmg::ParallelFor(box, 2, [=] AMREX_GPU_DEVICE (int i, int j, int, int n) noexcept
        {
            res(i,j,0,n) = residual1(i, j, n, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,n),
                                     acf(i,j,0), facx, facy);
        });
    } else {
        hpmg::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            res(i,j,0,0) = residual2r(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0),
                                      acf(i,j,0,0), acf(i,j,0,1), facx, facy);
            res(i,j,0,1) = residual2i(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,1),
                                      acf(i,j,0,0), acf(i,j,0,1), facx, facy);
        });
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void gs1 (int i, int j, int n, int ilo, int jlo, int ihi, int jhi,
          Array4<Real> const& phi, Real rhs, Real acf, Real facx, Real facy)
{
    Real lap;
    Real c0 = -(acf+Real(2.)*(facx+facy));
    if (i == ilo) {
        lap = facx * Real(4./3.)*phi(i+1,j,0,n);
        c0 -= Real(2.)*facx;
    } else if (i == ihi) {
        lap = facx * Real(4./3.)*phi(i-1,j,0,n);
        c0 -= Real(2.)*facx;
    } else {
        lap = facx * (phi(i-1,j,0,n) + phi(i+1,j,0,n));
    }
    if (j == jlo) {
        lap += facy * Real(4./3.)*phi(i,j+1,0,n);
        c0 -= Real(2.)*facy;
    } else if (j == jhi) {
        lap += facy * Real(4./3.)*phi(i,j-1,0,n);
        c0 -= Real(2.)*facy;
    } else {
        lap += facy * (phi(i,j-1,0,n) + phi(i,j+1,0,n));
    }
    phi(i,j,0,n) = (rhs - lap) / c0;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void gs2 (int i, int j, int ilo, int jlo, int ihi, int jhi,
          Array4<Real> const& phi, Real rhs_r, Real rhs_i,
          Real ar, Real ai, Real facx, Real facy)
{
    Real lap[2];
    Real c0 = Real(-2.)*(facx+facy);
    if (i == ilo) {
        lap[0] = facx * Real(4./3.)*phi(i+1,j,0,0);
        lap[1] = facx * Real(4./3.)*phi(i+1,j,0,1);
        c0 -= Real(2.)*facx;
    } else if (i == ihi) {
        lap[0] = facx * Real(4./3.)*phi(i-1,j,0,0);
        lap[1] = facx * Real(4./3.)*phi(i-1,j,0,1);
        c0 -= Real(2.)*facx;
    } else {
        lap[0] = facx * (phi(i-1,j,0,0) + phi(i+1,j,0,0));
        lap[1] = facx * (phi(i-1,j,0,1) + phi(i+1,j,0,1));
    }
    if (j == jlo) {
        lap[0] += facy * Real(4./3.)*phi(i,j+1,0,0);
        lap[1] += facy * Real(4./3.)*phi(i,j+1,0,1);
        c0 -= Real(2.)*facy;
    } else if (j == jhi) {
        lap[0] += facy * Real(4./3.)*phi(i,j-1,0,0);
        lap[1] += facy * Real(4./3.)*phi(i,j-1,0,1);
        c0 -= Real(2.)*facy;
    } else {
        lap[0] += facy * (phi(i,j-1,0,0) + phi(i,j+1,0,0));
        lap[1] += facy * (phi(i,j-1,0,1) + phi(i,j+1,0,1));
    }
    Real c[2] = {c0-ar, -ai};
    Real cmag = Real(1.)/(c[0]*c[0] + c[1]*c[1]);
    phi(i,j,0,0) = ((rhs_r-lap[0])*c[0] + (rhs_i-lap[1])*c[1]) * cmag;
    phi(i,j,0,1) = ((rhs_i-lap[1])*c[0] - (rhs_r-lap[0])*c[1]) * cmag;
}

void gsrb (int icolor, Box const& box, Array4<Real> const& phi,
           Array4<Real const> const& rhs, Array4<Real const> const& acf,
           Real dx, Real dy, int system_type)
{
    int const ilo = box.smallEnd(0);
    int const jlo = box.smallEnd(1);
    int const ihi = box.bigEnd(0);
    int const jhi = box.bigEnd(1);
    Real facx = Real(1.)/(dx*dx);
    Real facy = Real(1.)/(dy*dy);
    if (system_type == 1) {
        hpmg::ParallelFor(box, 2, [=] AMREX_GPU_DEVICE (int i, int j, int, int n) noexcept
        {
            if ((i+j+icolor)%2 == 0) {
                gs1(i, j, n, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,n), acf(i,j,0), facx, facy);
            }
        });
    } else {
        hpmg::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            if ((i+j+icolor)%2 == 0) {
                gs2(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), rhs(i,j,0,1),
                    acf(i,j,0,0), acf(i,j,0,1), facx, facy);
            }
        });
    }
}

void restriction (Box const& box, Array4<Real> const& crse, Array4<Real const> const& fine)
{
    hpmg::ParallelFor(box, 2, [=] AMREX_GPU_DEVICE (int i, int j, int, int n) noexcept
    {
        crse(i,j,0,n) = Real(0.25)*(fine(2*i  ,2*j  ,0,n) +
                                    fine(2*i+1,2*j  ,0,n) +
                                    fine(2*i  ,2*j+1,0,n) +
                                    fine(2*i+1,2*j+1,0,n));

    });
}

void interpolation (Box const& box, Array4<Real> const& fine, Array4<Real const> const& crse)
{
    hpmg::ParallelFor(box, 2, [=] AMREX_GPU_DEVICE (int i, int j, int, int n) noexcept
    {
        int ic = amrex::coarsen(i,2);
        int jc = amrex::coarsen(j,2);
        fine(i,j,0,n) += crse(ic,jc,0,n);
    });
}

#if defined(AMREX_USE_GPU)

#if defined(AMREX_USE_DPCPP)
#define HPMG_SYNCTHREADS item.barrier(sycl::access::fence_space::global_and_local)
#else
#define HPMG_SYNCTHREADS __syncthreads()
#endif

template <int NS, typename FGS, typename FRES>
void bottomsolve_gpu (Real dx0, Real dy0, Array4<Real> const* acf,
                      Array4<Real> const* res, Array4<Real> const* cor,
                      Array4<Real> const* rescor, int nlevs,
                      FGS&& fgs, FRES&& fres)
{
    static_assert(n_cell_single*n_cell_single <= 1024, "n_cell_single is too big");
#if defined(AMREX_USE_DPCPP)
    amrex::launch(1, 1024, Gpu::gpuStream(),
    [=] (sycl::nd_item<1> const& item) noexcept
#else
    amrex::launch_global<1024><<<1, 1024, 0, Gpu::gpuStream()>>>(
    [=] AMREX_GPU_DEVICE () noexcept
#endif
    {
        Real facx = Real(1.)/(dx0*dx0);
        Real facy = Real(1.)/(dy0*dy0);
        int lenx = cor[0].end.x - cor[0].begin.x;
        int leny = cor[0].end.y - cor[0].begin.y;
        int ncells = lenx*leny;
#if defined(AMREX_USE_DPCPP)
        const int icell = item.get_local_linear_id();
#else
        const int icell = threadIdx.x;
#endif

        for (int ilev = 0; ilev < nlevs-1; ++ilev) {
            if (icell < ncells) {
                cor[ilev].p[icell] = Real(0.);
                cor[ilev].p[icell+ncells] = Real(0.);
            }
            HPMG_SYNCTHREADS;

            for (int is = 0; is < 4; ++is) {
                if (icell < ncells) {
                    int j = icell /   lenx;
                    int i = icell - j*lenx;
                    j += cor[ilev].begin.y;
                    i += cor[ilev].begin.x;
                    if ((i+j+is)%2 == 0) {
                        fgs(i, j,
                            cor[ilev].begin.x, cor[ilev].begin.y,
                            cor[ilev].end.x-1, cor[ilev].end.y-1,
                            cor[ilev],
                            res[ilev](i,j,0,0),
                            res[ilev](i,j,0,1),
                            acf[ilev], facx, facy);
                    }
                }
                HPMG_SYNCTHREADS;
            }

            if (icell < ncells) {
                int j = icell /   lenx;
                int i = icell - j*lenx;
                j += cor[ilev].begin.y;
                i += cor[ilev].begin.x;
                fres(i, j,
                     rescor[ilev](i,j,0,0),
                     rescor[ilev](i,j,0,1),
                     cor[ilev].begin.x, cor[ilev].begin.y,
                     cor[ilev].end.x-1, cor[ilev].end.y-1,
                     cor[ilev],
                     res[ilev](i,j,0,0),
                     res[ilev](i,j,0,1),
                     acf[ilev], facx, facy);
            }
            HPMG_SYNCTHREADS;

            lenx = cor[ilev+1].end.x - cor[ilev+1].begin.x;
            leny = cor[ilev+1].end.y - cor[ilev+1].begin.y;
            ncells = lenx*leny;
            if (icell < ncells) {
                int j = icell /   lenx;
                int i = icell - j*lenx;
                j += cor[ilev+1].begin.y;
                i += cor[ilev+1].begin.x;
                for (int n = 0; n < 2; ++n) {
                    res[ilev+1](i,j,0,n) = Real(0.25)*(rescor[ilev](2*i  ,2*j  ,0,n) +
                                                       rescor[ilev](2*i+1,2*j  ,0,n) +
                                                       rescor[ilev](2*i  ,2*j+1,0,n) +
                                                       rescor[ilev](2*i+1,2*j+1,0,n));
                }
            }
            HPMG_SYNCTHREADS;

            facx *= Real(0.25);
            facy *= Real(0.25);
        }

        // bottom
        {
            const int ilev = nlevs-1;
            if (icell < ncells) {
                cor[ilev].p[icell] = Real(0.);
                cor[ilev].p[icell+ncells] = Real(0.);
            }
            HPMG_SYNCTHREADS;

            for (int is = 0; is < NS; ++is) {
                if (icell < ncells) {
                    int j = icell /   lenx;
                    int i = icell - j*lenx;
                    j += cor[ilev].begin.y;
                    i += cor[ilev].begin.x;
                    if ((i+j+is)%2 == 0) {
                        fgs(i, j,
                            cor[ilev].begin.x, cor[ilev].begin.y,
                            cor[ilev].end.x-1, cor[ilev].end.y-1,
                            cor[ilev],
                            res[ilev](i,j,0,0),
                            res[ilev](i,j,0,1),
                            acf[ilev], facx, facy);
                    }
                }
                HPMG_SYNCTHREADS;
            }
        }

        for (int ilev = nlevs-2; ilev >=0; --ilev) {
            lenx = cor[ilev].end.x - cor[ilev].begin.x;
            leny = cor[ilev].end.y - cor[ilev].begin.y;
            ncells = lenx*leny;
            facx *= Real(4.);
            facy *= Real(4.);

            if (icell < ncells) {
                int j = icell /   lenx;
                int i = icell - j*lenx;
                j += cor[ilev].begin.y;
                i += cor[ilev].begin.x;
                int ic = amrex::coarsen(i,2);
                int jc = amrex::coarsen(j,2);
                cor[ilev](i,j,0,0) += cor[ilev+1](ic,jc,0,0);
                cor[ilev](i,j,0,1) += cor[ilev+1](ic,jc,0,1);
            }

            for (int is = 0; is < 4; ++is) {
                HPMG_SYNCTHREADS;
                if (icell < ncells) {
                    int j = icell /   lenx;
                    int i = icell - j*lenx;
                    j += cor[ilev].begin.y;
                    i += cor[ilev].begin.x;
                    if ((i+j+is)%2 == 0) {
                        fgs(i, j,
                            cor[ilev].begin.x, cor[ilev].begin.y,
                            cor[ilev].end.x-1, cor[ilev].end.y-1,
                            cor[ilev],
                            res[ilev](i,j,0,0),
                            res[ilev](i,j,0,1),
                            acf[ilev], facx, facy);
                    }
                }
            }
        }
    });
}

#endif // AMREX_USE_GPU

} // namespace {}

MultiGrid::MultiGrid (Geometry const& geom)
    : m_dx(geom.CellSize(0)), m_dy(geom.CellSize(1))
{
    Box const& a_domain = geom.Domain();
    AMREX_ALWAYS_ASSERT(a_domain.length(2) == 1 && a_domain.cellCentered());

    m_domain.push_back(amrex::makeSlab(a_domain, 2, 0));
    for (int i = 0; i < 30; ++i) {
        if (m_domain.back().coarsenable(IntVect(2,2,1), IntVect(2,2,1))) {
            m_domain.push_back(amrex::coarsen(m_domain.back(),IntVect(2,2,1)));
        } else {
            break;
        }
    }
    m_max_level = m_domain.size()-1;
#if defined(AMREX_USE_GPU)
    auto r = std::find_if(std::begin(m_domain), std::end(m_domain),
                          [=] (Box const& b) -> bool
                              { return b.numPts() <= n_cell_single*n_cell_single; });
    m_single_block_level_begin = std::distance(std::begin(m_domain), r);
    m_single_block_level_begin = std::max(1, m_single_block_level_begin);
#else
    m_single_block_level_begin = m_max_level;
#endif

    m_num_mg_levels = m_max_level+1;
    m_num_single_block_levels = m_num_mg_levels - m_single_block_level_begin;

    if (m_num_single_block_levels > 0) {
        m_h_array4.reserve(nfabvs*m_num_single_block_levels);
    }

    m_acf.reserve(m_num_mg_levels);
    for (int ilev = 0; ilev < m_num_mg_levels; ++ilev) {
        m_acf.emplace_back(m_domain[ilev], 2);
        if (ilev >= m_single_block_level_begin) {
            m_h_array4.push_back(m_acf[ilev].array());
        }
    }

    m_res.reserve(m_num_mg_levels);
    for (int ilev = 0; ilev < m_num_mg_levels; ++ilev) {
        m_res.emplace_back(m_domain[ilev], 2);
        if (ilev >= m_single_block_level_begin) {
            m_h_array4.push_back(m_res[ilev].array());
        }
    }

    m_cor.reserve(m_num_mg_levels);
    for (int ilev = 0; ilev < m_num_mg_levels; ++ilev) {
        m_cor.emplace_back(m_domain[ilev], 2);
        if (ilev >= m_single_block_level_begin) {
            m_h_array4.push_back(m_cor[ilev].array());
        }
    }

    m_rescor.reserve(m_num_mg_levels);
    for (int ilev = 0; ilev < m_num_mg_levels; ++ilev) {
        m_rescor.emplace_back(m_domain[ilev], 2);
        if (ilev >= m_single_block_level_begin) {
            m_h_array4.push_back(m_rescor[ilev].array());
        }
    }

    if (!m_h_array4.empty()) {
        m_d_array4.resize(m_h_array4.size());
        Gpu::copyAsync(Gpu::hostToDevice, m_h_array4.begin(), m_h_array4.end(),
                       m_d_array4.begin());
        m_acf_a = m_d_array4.data();
        m_res_a = m_acf_a + m_num_single_block_levels;
        m_cor_a = m_res_a + m_num_single_block_levels;
        m_rescor_a = m_cor_a + m_num_single_block_levels;
    }
}

void
MultiGrid::solve1 (FArrayBox& a_sol, FArrayBox const& a_rhs, FArrayBox const& a_acf,
                   Real const tol_rel, Real const tol_abs, int const nummaxiter,
                   int const verbose)
{
    HIPACE_PROFILE("hpmg::MultiGrid::solve1()");
    m_system_type = 1;

    AMREX_ALWAYS_ASSERT(amrex::makeSlab(a_acf.box(),2,0).contains(m_domain.front()));
    FArrayBox afab(amrex::makeSlab(a_acf.box(), 2, 0), 1, a_acf.dataPtr());

    Array2<amrex::Real> const& array_m_acf = m_acf[0].array();
    Array2<const amrex::Real> const& array_a_acf = a_acf.const_array();
    hpmg::ParallelFor(m_acf[0].box(),
        [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            array_m_acf(i,j) = array_a_acf(i,j);
        });

    average_down_acoef();

    solve_doit(a_sol, a_rhs, tol_rel, tol_abs, nummaxiter, verbose);
}

void
MultiGrid::solve_doit (FArrayBox& a_sol, FArrayBox const& a_rhs,
                       Real const tol_rel, Real const tol_abs, int const nummaxiter,
                       int const verbose)
{
    AMREX_ALWAYS_ASSERT(amrex::makeSlab(a_rhs.box(),2,0).contains(m_domain.front()) &&
                        a_sol.nComp() >= 2 && a_rhs.nComp() >= 2);

    m_sol = FArrayBox(amrex::makeSlab(a_sol.box(), 2, 0), 2, a_sol.dataPtr());
    m_rhs = FArrayBox(amrex::makeSlab(a_rhs.box(), 2, 0), 2, a_rhs.dataPtr());

    compute_residual(m_domain[0], m_res[0].array(), m_sol.array(),
                     m_rhs.const_array(), m_acf[0].const_array(), m_dx, m_dy,
                     m_system_type);

    Real resnorm0, rhsnorm0;
    {
        ReduceOps<ReduceOpMax,ReduceOpMax> reduce_op;
        ReduceData<Real,Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        const auto& array_res = m_res[0].const_array();
        const auto& array_rhs = m_rhs.const_array();
        reduce_op.eval(m_domain[0], 2, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int, int n) noexcept -> ReduceTuple
            {
                return {std::abs(array_res(i,j,0,n)), std::abs(array_rhs(i,j,0,n))};
            });

        auto hv = reduce_data.value(reduce_op);
        resnorm0 = amrex::get<0>(hv);
        rhsnorm0 = amrex::get<1>(hv);
    }
    if (verbose >= 1) {
        amrex::Print() << "hpmg: Initial rhs               = " << rhsnorm0 << "\n"
                       << "hpmg: Initial residual (resid0) = " << resnorm0 << "\n";
    }

    Real max_norm;
    std::string norm_name;
    if (rhsnorm0 >= resnorm0) {
        norm_name = "bnorm";
        max_norm = rhsnorm0;
    } else {
        norm_name = "resid0";
        max_norm = resnorm0;
    }
    const Real res_target = std::max(tol_abs, std::max(tol_rel,Real(1.e-16))*max_norm);

    if (resnorm0 <= res_target) {
        if (verbose >= 1) {
            amrex::Print() << "hpmg: No iterations needed\n";
        }
    } else {
        Real norminf = 0.;
        bool converged = true;

        for (int iter = 0; iter < nummaxiter; ++iter) {

            converged = false;

            vcycle();

            compute_residual(m_domain[0], m_res[0].array(), m_sol.array(),
                             m_rhs.const_array(), m_acf[0].const_array(), m_dx, m_dy,
                             m_system_type);

            Real const* pres0 = m_res[0].dataPtr();
            norminf = Reduce::Max<Real>(m_domain[0].numPts()*2,
                                        [=] AMREX_GPU_DEVICE (Long i) -> Real
                                        {
                                            return std::abs(pres0[i]);
                                        });
            if (verbose >= 2) {
                amrex::Print() << "hpmg: Iteration " << std::setw(3) << iter+1 << " resid/"
                               << norm_name << " = " << norminf/max_norm << "\n";
            }

            converged = (norminf <= res_target);
            if (converged) {
                if (verbose >= 1) {
                    amrex::Print() << "hpmg: Final Iter. " << iter+1
                                   << " resid, resid/" << norm_name << " = "
                                   << norminf << ", " << norminf/max_norm << "\n";
                }
                break;
            } else if (norminf > Real(1.e20)*max_norm) {
                if (verbose > 0) {
                    amrex::Print() << "hpmg: Failing to converge after " << iter+1 << " iterations."
                                   << " resid, resid/" << norm_name << " = "
                                   << norminf << ", " << norminf/max_norm << "\n";
                  }
                  amrex::Abort("hpmg failing so lets stop here");
            }
        }

        if (!converged) {
            if (verbose > 0) {
                amrex::Print() << "hpmg: Failed to converge after " << nummaxiter << " iterations."
                               << " resid, resid/" << norm_name << " = "
                               << norminf << ", " << norminf/max_norm << "\n";
            }
            amrex::Abort("hpmg failed");
        }
    }
}

void
MultiGrid::vcycle ()
{
#if defined(AMREX_USE_CUDA)
    const int igraph = m_system_type-1;
    bool& graph_created = m_cuda_graph_vcycle_created[igraph];
    cudaGraph_t& graph = m_cuda_graph_vcycle[igraph];
    cudaGraphExec_t& graph_exe = m_cuda_graph_exe_vcycle[igraph];
    if (!graph_created) {
    cudaStreamBeginCapture(Gpu::gpuStream(), cudaStreamCaptureModeGlobal);
#endif

    for (int ilev = 0; ilev < m_single_block_level_begin; ++ilev) {
        Real * pcor = m_cor[ilev].dataPtr();
        hpmg::ParallelFor(m_domain[ilev].numPts()*2,
                          [=] AMREX_GPU_DEVICE (Long i) noexcept { pcor[i] = Real(0.); });

        Real fac = static_cast<Real>(1 << ilev);
        Real dx = m_dx * fac;
        Real dy = m_dy * fac;
        for (int is = 0; is < 4; ++is) {
            gsrb(is, m_domain[ilev], m_cor[ilev].array(),
                 m_res[ilev].const_array(), m_acf[ilev].const_array(), dx, dy,
                 m_system_type);
        }

        // rescor = res - L(cor)
        compute_residual(m_domain[ilev], m_rescor[ilev].array(), m_cor[ilev].array(),
                         m_res[ilev].const_array(), m_acf[ilev].const_array(), dx, dy,
                         m_system_type);

        // res[ilev+1] = R(rescor[ilev])
        restriction(m_domain[ilev+1], m_res[ilev+1].array(), m_rescor[ilev].const_array());
    }

    bottomsolve();

    for (int ilev = m_single_block_level_begin-1; ilev >= 0; --ilev) {
        // cor[ilev] += I(cor[ilev+1])
        interpolation(m_domain[ilev], m_cor[ilev].array(), m_cor[ilev+1].const_array());

        Real fac = static_cast<Real>(1 << ilev);
        Real dx = m_dx * fac;
        Real dy = m_dy * fac;
        for (int is = 0; is < 4; ++is) {
            gsrb(is, m_domain[ilev], m_cor[ilev].array(),
                 m_res[ilev].const_array(), m_acf[ilev].const_array(), dx, dy,
                 m_system_type);
        }
    }

#if defined(AMREX_USE_CUDA)
    cudaStreamEndCapture(Gpu::gpuStream(), &graph);
    cudaGraphInstantiate(&graph_exe, graph, NULL, NULL, 0);
    graph_created = true;
    }
    cudaGraphLaunch(graph_exe, Gpu::gpuStream());
#endif

    auto const& sol = m_sol.array();
    auto const& cor = m_cor[0].const_array();
    hpmg::ParallelFor(m_domain[0], 2, [=] AMREX_GPU_DEVICE (int i, int j, int, int n) noexcept
    {
        sol(i,j,0,n) += cor(i,j,0,n);
    });
}

void
MultiGrid::bottomsolve ()
{
    constexpr int nsweeps = 16;
    Real fac = static_cast<Real>(1 << m_single_block_level_begin);
    Real dx0 = m_dx * fac;
    Real dy0 = m_dy * fac;
#if defined(AMREX_USE_GPU)
    Array4<amrex::Real> const* acf = m_acf_a;
    Array4<amrex::Real> const* res = m_res_a;
    Array4<amrex::Real> const* cor = m_cor_a;
    Array4<amrex::Real> const* rescor = m_rescor_a;
    int nlevs = m_num_single_block_levels;

    if (m_system_type == 1) {
        bottomsolve_gpu<nsweeps>(dx0, dy0, acf, res, cor, rescor, nlevs,
            [=] AMREX_GPU_DEVICE (int i, int j, int ilo, int jlo, int ihi, int jhi,
                                  Array4<Real> const& phi, Real rhs0, Real rhs1,
                                  Array4<Real> const& acf, Real facx, Real facy)
            {
                Real a = acf(i,j,0);
                gs1(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs0, a, facx, facy);
                gs1(i, j, 1, ilo, jlo, ihi, jhi, phi, rhs1, a, facx, facy);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, Real& res0, Real& res1,
                                  int ilo, int jlo, int ihi, int jhi,
                                  Array4<Real> const& phi, Real rhs0, Real rhs1,
                                  Array4<Real> const& acf, Real facx, Real facy)
            {
                Real a = acf(i,j,0);
                res0 = residual1(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs0, a, facx, facy);
                res1 = residual1(i, j, 1, ilo, jlo, ihi, jhi, phi, rhs1, a, facx, facy);
            });
    } else {
        bottomsolve_gpu<nsweeps>(dx0, dy0, acf, res, cor, rescor, nlevs,
            [=] AMREX_GPU_DEVICE (int i, int j, int ilo, int jlo, int ihi, int jhi,
                                  Array4<Real> const& phi, Real rhs0, Real rhs1,
                                  Array4<Real> const& acf, Real facx, Real facy)
            {
                Real ar = acf(i,j,0,0);
                Real ai = acf(i,j,0,1);
                gs2(i, j, ilo, jlo, ihi, jhi, phi, rhs0, rhs1, ar, ai, facx, facy);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, Real& res0, Real& res1,
                                  int ilo, int jlo, int ihi, int jhi,
                                  Array4<Real> const& phi, Real rhs_r, Real rhs_i,
                                  Array4<Real> const& acf, Real facx, Real facy)
            {
                Real ar = acf(i,j,0,0);
                Real ai = acf(i,j,0,1);
                res0 = residual2r(i, j, ilo, jlo, ihi, jhi, phi, rhs_r, ar, ai, facx, facy);
                res1 = residual2i(i, j, ilo, jlo, ihi, jhi, phi, rhs_i, ar, ai, facx, facy);
            });
    }
#else
    const int ilev = m_single_block_level_begin;
    m_cor[ilev].setVal(Real(0.));
    for (int is = 0; is < nsweeps; ++is) {
        gsrb(is, m_domain[ilev], m_cor[ilev].array(),
             m_res[ilev].const_array(), m_acf[ilev].const_array(), dx0, dy0,
             m_system_type);
    }
#endif
}

void
MultiGrid::average_down_acoef ()
{
    const int ncomp = (m_system_type == 1) ? 1 : 2;
#if defined(AMREX_USE_CUDA)
    const int igraph = m_system_type-1;
    bool& graph_created = m_cuda_graph_acf_created[igraph];
    cudaGraph_t& graph = m_cuda_graph_acf[igraph];
    cudaGraphExec_t& graph_exe = m_cuda_graph_exe_acf[igraph];
    if (!graph_created) {
    cudaStreamBeginCapture(Gpu::gpuStream(), cudaStreamCaptureModeGlobal);
#endif

    for (int ilev = 1; ilev <= m_single_block_level_begin; ++ilev) {
        auto const& crse = m_acf[ilev].array();
        auto const& fine = m_acf[ilev-1].const_array();
        hpmg::ParallelFor(m_domain[ilev], ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int, int n) noexcept
        {
            crse(i,j,0,n) = Real(0.25)*(fine(2*i  ,2*j  ,0,n) +
                                        fine(2*i+1,2*j  ,0,n) +
                                        fine(2*i  ,2*j+1,0,n) +
                                        fine(2*i+1,2*j+1,0,n));
        });
    }

#if defined(AMREX_USE_GPU)
    if (m_num_single_block_levels > 1) {
        Array4<Real> const* acf = m_acf_a;
        int nlevels = m_num_single_block_levels;

#if defined(AMREX_USE_DPCPP)
        amrex::launch(1, 1024, Gpu::gpuStream(),
        [=] (sycl::nd_item<1> const& item) noexcept
#else
        amrex::launch_global<1024><<<1, 1024, 0, Gpu::gpuStream()>>>(
        [=] AMREX_GPU_DEVICE () noexcept
#endif
        {
            for (int ilev = 1; ilev < nlevels; ++ilev) {
                const int lenx = acf[ilev].end.x - acf[ilev].begin.x;
                const int leny = acf[ilev].end.y - acf[ilev].begin.y;
                const int ncells = lenx*leny;
#if defined(AMREX_USE_DPCPP)
                for (int icell = item.get_local_range(0)*item.get_group_linear_id()
                         + item.get_local_linear_id(),
                         stride = item.get_local_range(0)*item.get_group_range(0);
#else
                for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
#endif
                     icell < ncells; icell += stride) {
                    int j = icell /   lenx;
                    int i = icell - j*lenx;
                    j += acf[ilev].begin.y;
                    i += acf[ilev].begin.x;
                    for (int n = 0; n < ncomp; ++n) {
                        acf[ilev](i,j,0,n) = Real(0.25)*(acf[ilev-1](2*i  ,2*j  ,0,n) +
                                                         acf[ilev-1](2*i+1,2*j  ,0,n) +
                                                         acf[ilev-1](2*i  ,2*j+1,0,n) +
                                                         acf[ilev-1](2*i+1,2*j+1,0,n));
                    }
                }
                HPMG_SYNCTHREADS;
            }
        });
    }
#endif

#if defined(AMREX_USE_CUDA)
    cudaStreamEndCapture(Gpu::gpuStream(), &graph);
    cudaGraphInstantiate(&graph_exe, graph, NULL, NULL, 0);
    graph_created = true;
    }
    cudaGraphLaunch(graph_exe, Gpu::gpuStream());
#endif
}

MultiGrid::~MultiGrid ()
{
#if defined(AMREX_USE_CUDA)
    for (int igraph = 0; igraph < m_num_system_types; ++igraph) {
        if (m_cuda_graph_acf_created[igraph]) {
            cudaGraphDestroy(m_cuda_graph_acf[igraph]);
            cudaGraphExecDestroy(m_cuda_graph_exe_acf[igraph]);
        }
        if (m_cuda_graph_vcycle_created[igraph]) {
            cudaGraphDestroy(m_cuda_graph_vcycle[igraph]);
            cudaGraphExecDestroy(m_cuda_graph_exe_vcycle[igraph]);
        }
    }
#endif
}

}
