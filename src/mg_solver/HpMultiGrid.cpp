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

Box valid_domain_box (Box const& domain)
{
    return domain.cellCentered() ? domain : amrex::grow(domain, IntVect(-1,-1,0));
}

template <typename T, typename U>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void restrict_cc (int i, int j, int n, Array4<T> const& crse, Array4<U> const& fine)
{
    crse(i,j,0,n) = Real(0.25)*(fine(2*i  ,2*j  ,0,n) +
                                fine(2*i+1,2*j  ,0,n) +
                                fine(2*i  ,2*j+1,0,n) +
                                fine(2*i+1,2*j+1,0,n));
}

template <typename T, typename U>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void restrict_nd (int i, int j, int n, Array4<T> const& crse, Array4<U> const& fine)
{
    crse(i,j,0,n) = Real(1./16.) * (fine(2*i-1,2*j-1,0,n) +
                           Real(2.)*fine(2*i  ,2*j-1,0,n) +
                                    fine(2*i+1,2*j-1,0,n) +
                           Real(2.)*fine(2*i-1,2*j  ,0,n) +
                           Real(4.)*fine(2*i  ,2*j  ,0,n) +
                           Real(2.)*fine(2*i+1,2*j  ,0,n) +
                                    fine(2*i-1,2*j+1,0,n) +
                           Real(2.)*fine(2*i  ,2*j+1,0,n) +
                                    fine(2*i+1,2*j+1,0,n));
}

template <typename T, typename U>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void interpadd_cc (int i, int j, int n, Array4<T> const& fine, Array4<U> const& crse)
{
    int ic = amrex::coarsen(i,2);
    int jc = amrex::coarsen(j,2);
    fine(i,j,0,n) += crse(ic,jc,0,n);
}

template <typename T, typename U>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void interpadd_nd (int i, int j, int n, Array4<T> const& fine, Array4<U> const& crse)
{
    int ic = amrex::coarsen(i,2);
    int jc = amrex::coarsen(j,2);
    bool i_is_odd = (ic*2 != i);
    bool j_is_odd = (jc*2 != j);
    if (i_is_odd && j_is_odd) {
        fine(i,j,0,n) += (crse(ic  ,jc  ,0,n) +
                          crse(ic+1,jc  ,0,n) +
                          crse(ic  ,jc+1,0,n) +
                          crse(ic+1,jc+1,0,n))*Real(0.25);
    } else if (i_is_odd) {
        fine(i,j,0,n) += (crse(ic  ,jc,0,n) +
                          crse(ic+1,jc,0,n))*Real(0.5);
    } else if (j_is_odd) {
        fine(i,j,0,n) += (crse(ic,jc  ,0,n) +
                          crse(ic,jc+1,0,n))*Real(0.5);
    } else {
        fine(i,j,0,n) += crse(ic,jc,0,n);
    }
}

// out of place version used before shared memory gsrb
template <typename T, typename U, typename V>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void interpcpy_cc (int i, int j, int n, Array4<T> const& fine_in, Array4<U> const& crse,
                Array4<V> const& fine_out)
{
    int ic = amrex::coarsen(i,2);
    int jc = amrex::coarsen(j,2);
    fine_out(i,j,0,n) = fine_in(i,j,0,n) + crse(ic,jc,0,n);
}

template <typename T, typename U, typename V>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void interpcpy_nd (int i, int j, int n, Array4<T> const& fine_in, Array4<U> const& crse,
                Array4<V> const& fine_out)
{
    int ic = amrex::coarsen(i,2);
    int jc = amrex::coarsen(j,2);
    bool i_is_odd = (ic*2 != i);
    bool j_is_odd = (jc*2 != j);
    if (i_is_odd && j_is_odd) {
        fine_out(i,j,0,n) = fine_in(i,j,0,n) + (crse(ic  ,jc  ,0,n) +
                                                crse(ic+1,jc  ,0,n) +
                                                crse(ic  ,jc+1,0,n) +
                                                crse(ic+1,jc+1,0,n))*Real(0.25);
    } else if (i_is_odd) {
        fine_out(i,j,0,n) = fine_in(i,j,0,n) + (crse(ic  ,jc,0,n) +
                                                crse(ic+1,jc,0,n))*Real(0.5);
    } else if (j_is_odd) {
        fine_out(i,j,0,n) = fine_in(i,j,0,n) + (crse(ic,jc  ,0,n) +
                                                crse(ic,jc+1,0,n))*Real(0.5);
    } else {
        fine_out(i,j,0,n) = fine_in(i,j,0,n) +  crse(ic,jc,0,n);
    }
}

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

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real residual3 (int i, int j, int n, int ilo, int jlo, int ihi, int jhi,
                Array4<Real> const& phi, Real rhs, Real facx, Real facy)
{
    Real lap = laplacian(i,j,n,ilo,jlo,ihi,jhi,phi,facx,facy);
    return rhs - lap;
}

#if !defined(AMREX_USE_CUDA) && !defined(AMREX_USE_HIP)
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
        hpmg::ParallelFor(to2D(valid_domain_box(box)),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            res(i,j,0,0) = residual1(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0),
                                     acf(i,j,0), facx, facy);
            res(i,j,0,1) = residual1(i, j, 1, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,1),
                                     acf(i,j,0), facx, facy);
        });
    } else if (system_type == 2) {
        hpmg::ParallelFor(to2D(valid_domain_box(box)),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            res(i,j,0,0) = residual2r(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0),
                                      acf(i,j,0,0), acf(i,j,0,1), facx, facy);
            res(i,j,0,1) = residual2i(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,1),
                                      acf(i,j,0,0), acf(i,j,0,1), facx, facy);
        });
    } else {
        hpmg::ParallelFor(to2D(valid_domain_box(box)),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            res(i,j,0,0) = residual3(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), facx, facy);
        });
    }
}
#endif

// is_cell_centered = true: supports both cell centered and node centered solves
// is_cell_centered = false: only supports node centered solves, with higher performance
template<bool is_cell_centered = true>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void gs1 (int i, int j, int n, int ilo, int jlo, int ihi, int jhi,
          Array4<Real> const& phi, Real rhs, Real acf, Real facx, Real facy)
{
    Real lap;
    Real c0 = -(acf+Real(2.)*(facx+facy));
    if (is_cell_centered && i == ilo) {
        lap = facx * Real(4./3.)*phi(i+1,j,0,n);
        c0 -= Real(2.)*facx;
    } else if (is_cell_centered && i == ihi) {
        lap = facx * Real(4./3.)*phi(i-1,j,0,n);
        c0 -= Real(2.)*facx;
    } else {
        lap = facx * (phi(i-1,j,0,n) + phi(i+1,j,0,n));
    }
    if (is_cell_centered && j == jlo) {
        lap += facy * Real(4./3.)*phi(i,j+1,0,n);
        c0 -= Real(2.)*facy;
    } else if (is_cell_centered && j == jhi) {
        lap += facy * Real(4./3.)*phi(i,j-1,0,n);
        c0 -= Real(2.)*facy;
    } else {
        lap += facy * (phi(i,j-1,0,n) + phi(i,j+1,0,n));
    }
    const Real c0_inv = Real(1.) / c0;
    phi(i,j,0,n) = (rhs - lap) * c0_inv;
}

// is_cell_centered = true: supports both cell centered and node centered solves
// is_cell_centered = false: only supports node centered solves, with higher performance
template<bool is_cell_centered = true>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void gs2 (int i, int j, int ilo, int jlo, int ihi, int jhi,
          Array4<Real> const& phi, Real rhs_r, Real rhs_i,
          Real ar, Real ai, Real facx, Real facy)
{
    Real lap[2];
    Real c0 = Real(-2.)*(facx+facy);
    if (is_cell_centered && i == ilo) {
        lap[0] = facx * Real(4./3.)*phi(i+1,j,0,0);
        lap[1] = facx * Real(4./3.)*phi(i+1,j,0,1);
        c0 -= Real(2.)*facx;
    } else if (is_cell_centered && i == ihi) {
        lap[0] = facx * Real(4./3.)*phi(i-1,j,0,0);
        lap[1] = facx * Real(4./3.)*phi(i-1,j,0,1);
        c0 -= Real(2.)*facx;
    } else {
        lap[0] = facx * (phi(i-1,j,0,0) + phi(i+1,j,0,0));
        lap[1] = facx * (phi(i-1,j,0,1) + phi(i+1,j,0,1));
    }
    if (is_cell_centered && j == jlo) {
        lap[0] += facy * Real(4./3.)*phi(i,j+1,0,0);
        lap[1] += facy * Real(4./3.)*phi(i,j+1,0,1);
        c0 -= Real(2.)*facy;
    } else if (is_cell_centered && j == jhi) {
        lap[0] += facy * Real(4./3.)*phi(i,j-1,0,0);
        lap[1] += facy * Real(4./3.)*phi(i,j-1,0,1);
        c0 -= Real(2.)*facy;
    } else {
        lap[0] += facy * (phi(i,j-1,0,0) + phi(i,j+1,0,0));
        lap[1] += facy * (phi(i,j-1,0,1) + phi(i,j+1,0,1));
    }
    Real c[2] = {c0-ar, -ai};
    Real cmag = Real(1.)/(c[0]*c[0] + c[1]*c[1]);
    c[0] *= cmag;
    c[1] *= cmag;
    phi(i,j,0,0) = (rhs_r-lap[0])*c[0] + (rhs_i-lap[1])*c[1];
    phi(i,j,0,1) = (rhs_i-lap[1])*c[0] - (rhs_r-lap[0])*c[1];
}

// is_cell_centered = true: supports both cell centered and node centered solves
// is_cell_centered = false: only supports node centered solves, with higher performance
template<bool is_cell_centered = true>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void gs3 (int i, int j, int n, int ilo, int jlo, int ihi, int jhi,
          Array4<Real> const& phi, Real rhs, Real facx, Real facy)
{
    Real lap;
    Real c0 = -Real(2.)*(facx+facy);
    if (is_cell_centered && i == ilo) {
        lap = facx * Real(4./3.)*phi(i+1,j,0,n);
        c0 -= Real(2.)*facx;
    } else if (is_cell_centered && i == ihi) {
        lap = facx * Real(4./3.)*phi(i-1,j,0,n);
        c0 -= Real(2.)*facx;
    } else {
        lap = facx * (phi(i-1,j,0,n) + phi(i+1,j,0,n));
    }
    if (is_cell_centered && j == jlo) {
        lap += facy * Real(4./3.)*phi(i,j+1,0,n);
        c0 -= Real(2.)*facy;
    } else if (is_cell_centered && j == jhi) {
        lap += facy * Real(4./3.)*phi(i,j-1,0,n);
        c0 -= Real(2.)*facy;
    } else {
        lap += facy * (phi(i,j-1,0,n) + phi(i,j+1,0,n));
    }
    const Real c0_inv = Real(1.) / c0;
    phi(i,j,0,n) = (rhs - lap) * c0_inv;
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
        hpmg::ParallelFor(to2D(valid_domain_box(box)),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            if ((i+j+icolor)%2 == 0) {
                gs1(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), acf(i,j,0), facx, facy);
                gs1(i, j, 1, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,1), acf(i,j,0), facx, facy);
            }
        });
    } else if (system_type == 2) {
        hpmg::ParallelFor(to2D(valid_domain_box(box)),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            if ((i+j+icolor)%2 == 0) {
                gs2(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), rhs(i,j,0,1),
                    acf(i,j,0,0), acf(i,j,0,1), facx, facy);
            }
        });
    } else {
        hpmg::ParallelFor(to2D(valid_domain_box(box)),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            if ((i+j+icolor)%2 == 0) {
                gs3(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), facx, facy);
            }
        });
    }
}

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

// do multiple gsrb iterations in GPU shared memory with many ghost cells
template<int system_type, bool zero_init, bool do_compute_residual, bool is_cell_centered>
void gsrb_shared (Box const& box, Array4<Real> const& phi_out, Array4<Real const> const& rhs,
                  Array4<Real const> const& acf, Array4<Real> const& res,
                  Array4<Real const> const& phi_in, Real dx, Real dy)
{
    constexpr int num_comps = MultiGrid::get_num_comps(system_type);
    constexpr int num_comps_acf = MultiGrid::get_num_comps_acf(system_type);
    constexpr int tilesize_x = 64;
    constexpr int tilesize_y = 32;
    constexpr int num_threads = tilesize_x * tilesize_y / 2; // every thread calculates two elements
    constexpr int tilesize_array_x = tilesize_x + 2; // add one ghost cell in each direction
    constexpr int tilesize_array_y = tilesize_y + 2;
    constexpr int num_cells_in_tile = tilesize_array_x * tilesize_array_y * num_comps;
    constexpr int niter = 4;
    constexpr int edge_offset = niter - !do_compute_residual;
    constexpr int final_tilesize_x = tilesize_x - 2*edge_offset; // add more ghost cells
    constexpr int final_tilesize_y = tilesize_y - 2*edge_offset;
    static_assert(system_type == 1 || system_type == 2 || system_type == 3,
        "gsrb_shared only supports system type 1, 2 or 3");

    // box for the bounds of the arrays
    int const ilo = box.smallEnd(0);
    int const jlo = box.smallEnd(1);
    int const ihi = box.bigEnd(0);
    int const jhi = box.bigEnd(1);
    Real facx = Real(1.)/(dx*dx);
    Real facy = Real(1.)/(dy*dy);

    // box for the bounds of the ParallelFor loop this kernel replaces
    const Box loop_box = valid_domain_box(box);
    int const ilo_loop = loop_box.smallEnd(0);
    int const jlo_loop = loop_box.smallEnd(1);
    int const ihi_loop = loop_box.bigEnd(0);
    int const jhi_loop = loop_box.bigEnd(1);
    const int num_blocks_x = (loop_box.length(0) + final_tilesize_x - 1)/final_tilesize_x;
    const int num_blocks_y = (loop_box.length(1) + final_tilesize_y - 1)/final_tilesize_y;
    amrex::Math::FastDivmodU64 num_blocks_divmod {static_cast<std::uint64_t>(num_blocks_x)};

    amrex::launch<num_threads>(num_blocks_x*num_blocks_y, amrex::Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE() noexcept
        {
            // allocate static shared memory
            __shared__ Real phi_ptr[num_cells_in_tile];

            std::uint64_t remainder = 0;
            const int iblock_y = num_blocks_divmod.divmod(remainder, blockIdx.x);
            const int iblock_x = remainder;

            const int tile_begin_x = iblock_x * final_tilesize_x - edge_offset - 1 + ilo_loop;
            const int tile_begin_y = iblock_y * final_tilesize_y - edge_offset - 1 + jlo_loop;

            const int tile_end_x = tile_begin_x + tilesize_array_x;
            const int tile_end_y = tile_begin_y + tilesize_array_y;

            // make Array4 reference shared memory tile
            Array4<Real> phi_shared(phi_ptr, {tile_begin_x, tile_begin_y, 0},
                                             {tile_end_x, tile_end_y, 1}, num_comps);

            if (zero_init) {
                for (int s = threadIdx.x; s < num_cells_in_tile; s+=blockDim.x) {
                    phi_ptr[s] = Real(0.);
                }
            } else {
                for (int s = threadIdx.x; s < tilesize_array_x*tilesize_array_y; s+=blockDim.x) {
                    int sy = s / tilesize_array_x;
                    int sx = s - sy * tilesize_array_x;
                    sx += tile_begin_x;
                    sy += tile_begin_y;
                    if (ilo_loop <= sx && sx <= ihi_loop &&
                        jlo_loop <= sy && sy <= jhi_loop) {
                        for (int n=0; n<num_comps; ++n) {
                            phi_shared(sx, sy, 0, n) = phi_in(sx, sy, 0, n);
                        }
                    } else {
                        for (int n=0; n<num_comps; ++n) {
                            phi_shared(sx, sy, 0, n) = Real(0.);
                        }
                    }
                }
            }

            int ithread_y = threadIdx.x / tilesize_x;
            const int ithread_x = threadIdx.x - ithread_y * tilesize_x;
            ithread_y *= 2;

            const int i = tile_begin_x + 1 + ithread_x;
            const int j = tile_begin_y + 1 + ithread_y;

            amrex::GpuArray<Real, num_comps> rhs_num[2] = {{}, {}};
            amrex::GpuArray<Real, num_comps_acf> acf_num[2] = {{}, {}};

            for (int nj=0; nj<=1; ++nj) {
                if (ilo_loop <= i && i <= ihi_loop &&
                    jlo_loop <= j+nj && j+nj <= jhi_loop) {
                    // load rhs and acf into registers to avoid memory accesses in the gs iterations
                    for (int n=0; n<num_comps; ++n) {
                        rhs_num[nj][n] = rhs(i, j+nj, 0, n);
                    }
                    for (int n=0; n<num_comps_acf; ++n) {
                        acf_num[nj][n] = acf(i, j+nj, 0, n);
                    }
                }
            }

            __syncthreads();

            for (int icolor=0; icolor<niter; ++icolor) {
                // Do 4 Gauss–Seidel iterations.
                // Every thread updates elements (i,j) and (i,j+1). When updating the elements
                // in the checkerboard pattern every thread can pick the correct element instead
                // of every second thread doing nothing.
                // Note that this makes the memory access of phi_shared non coalesced,
                // this is ok for shared memory.
                const int shift = (i + j + icolor) & 1;
                const int j_loc = j + shift;
                const auto rhs_loc = shift ? rhs_num[1] : rhs_num[0];
                const auto acf_loc = shift ? acf_num[1] : acf_num[0];
                if (ilo_loop <= i && i <= ihi_loop &&
                    jlo_loop <= j_loc && j_loc <= jhi_loop) {
                    if (system_type == 1) {
                        gs1<is_cell_centered>(i, j_loc, 0, ilo, jlo, ihi, jhi, phi_shared,
                            rhs_loc[0], acf_loc[0], facx, facy);
                        gs1<is_cell_centered>(i, j_loc, 1, ilo, jlo, ihi, jhi, phi_shared,
                            rhs_loc[1], acf_loc[0], facx, facy);
                    } else if (system_type == 2) {
                        gs2<is_cell_centered>(i, j_loc, ilo, jlo, ihi, jhi, phi_shared,
                            rhs_loc[0], rhs_loc[1], acf_loc[0], acf_loc[1], facx, facy);
                    } else if (system_type == 3) {
                        gs3<is_cell_centered>(i, j_loc, 0, ilo, jlo, ihi, jhi, phi_shared,
                            rhs_loc[0], facx, facy);
                    }
                }
                __syncthreads();
            }

            for (int nj=0; nj<=1; ++nj) {
                if (ilo_loop <= i && i <= ihi_loop &&
                    jlo_loop <= j+nj && j+nj <= jhi_loop &&
                    edge_offset <= ithread_x && ithread_x < tilesize_x - edge_offset &&
                    edge_offset <= ithread_y+nj && ithread_y+nj < tilesize_y - edge_offset) {
                    // store results in global memory but only in the shrunken box
                    // where the result is correct

                    if (do_compute_residual) {
                        // fuse with compute_residual kernel and reuse phi_shared
                        // at the cost of one extra ghost cell in each direction
                        if (system_type == 1) {
                            res(i, j+nj, 0, 0) = residual1(
                                                    i, j+nj, 0, ilo, jlo, ihi, jhi, phi_shared,
                                                    rhs_num[nj][0], acf_num[nj][0], facx, facy);
                            res(i, j+nj, 0, 1) = residual1(
                                                    i, j+nj, 1, ilo, jlo, ihi, jhi, phi_shared,
                                                    rhs_num[nj][1], acf_num[nj][0], facx, facy);
                        } else if (system_type == 2) {
                            res(i, j+nj, 0, 0) = residual2r(
                                                    i, j+nj, ilo, jlo, ihi, jhi, phi_shared,
                                                    rhs_num[nj][0], acf_num[nj][0], acf_num[nj][1],
                                                    facx, facy);
                            res(i, j+nj, 0, 1) = residual2i(
                                                    i, j+nj, ilo, jlo, ihi, jhi, phi_shared,
                                                    rhs_num[nj][1], acf_num[nj][0], acf_num[nj][1],
                                                    facx, facy);
                        } else if (system_type == 3) {
                            res(i, j+nj, 0, 0) = residual3(
                                                    i, j+nj, 0, ilo, jlo, ihi, jhi, phi_shared,
                                                    rhs_num[nj][0], facx, facy);
                        }
                    }

                    for (int n=0; n<num_comps; ++n) {
                        phi_out(i, j+nj, 0, n) = phi_shared(i, j+nj, 0, n);
                    }
                }
            }
        });
}

#endif

template<bool zero_init, bool do_compute_residual>
void gsrb_4_residual (int system_type, Box const& box,
                      Array4<Real> const& phi_out,
                      Array4<Real const> const& rhs,
                      Array4<Real const> const& acf,
                      Array4<Real> const& res,
                      Array4<Real const> const& phi_in,
                      Real dx, Real dy)
{
    // Start at either phi_in or zero depending on if zero_init is set.
    // Compute 4 gsrb iterations using rhs and acf and store the result in phi_out.
    // If do_compute_residual is set, store the residual in res.
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    if (system_type == 1) {
        if (box.cellCentered()) {
            gsrb_shared<1, zero_init, do_compute_residual, true>(
                box, phi_out, rhs, acf, res, phi_in, dx, dy);
        } else {
            gsrb_shared<1, zero_init, do_compute_residual, false>(
                box, phi_out, rhs, acf, res, phi_in, dx, dy);
        }
    } else if (system_type == 2) {
        if (box.cellCentered()) {
            gsrb_shared<2, zero_init, do_compute_residual, true>(
                box, phi_out, rhs, acf, res, phi_in, dx, dy);
        } else {
            gsrb_shared<2, zero_init, do_compute_residual, false>(
                box, phi_out, rhs, acf, res, phi_in, dx, dy);
        }
    } else if (system_type == 3) {
        if (box.cellCentered()) {
            gsrb_shared<3, zero_init, do_compute_residual, true>(
                box, phi_out, rhs, acf, res, phi_in, dx, dy);
        } else {
            gsrb_shared<3, zero_init, do_compute_residual, false>(
                box, phi_out, rhs, acf, res, phi_in, dx, dy);
        }
    }
#else
    if (zero_init) {
        Real * pcor_out = phi_out.dataPtr();
        hpmg::ParallelFor(box.numPts()*MultiGrid::get_num_comps(system_type),
            [=] AMREX_GPU_DEVICE (Long i) noexcept { pcor_out[i] = Real(0.); });
    } else {
        const amrex::Box valid_domain = valid_domain_box(box);
        hpmg::ParallelFor(to2D(box), MultiGrid::get_num_comps(system_type),
            [=] AMREX_GPU_DEVICE (int i, int j, int n) noexcept {
                if (valid_domain.contains(i,j,0)) {
                    phi_out(i,j,0,n) = phi_in(i,j,0,n);
                } else {
                    phi_out(i,j,0,n) = Real(0.);
                }
            });
    }

    for (int is = 0; is < 4; ++is) {
        gsrb(is, box, phi_out, rhs, acf, dx, dy, system_type);
    }

    if (do_compute_residual) {
        compute_residual(box, res, phi_out, rhs, acf, dx, dy, system_type);
    }
#endif
}

void restriction (Box const& box, Array4<Real> const& crse, Array4<Real const> const& fine,
                  int num_comps)
{
    if (box.cellCentered()) {
        hpmg::ParallelFor(to2D(box), num_comps, [=] AMREX_GPU_DEVICE (int i, int j, int n) noexcept
        {
            restrict_cc(i,j,n,crse,fine);
        });
    } else {
        hpmg::ParallelFor(to2D(valid_domain_box(box)), num_comps,
        [=] AMREX_GPU_DEVICE (int i, int j, int n) noexcept
        {
            restrict_nd(i,j,n,crse,fine);
        });
    }
}

void interpolation_outofplace (Box const& box, Array4<Real const> const& fine_in,
                               Array4<Real const> const& crse, Array4<Real> const& fine_out,
                               int num_comps)
{
    if (box.cellCentered()) {
        hpmg::ParallelFor(to2D(box), num_comps, [=] AMREX_GPU_DEVICE (int i, int j, int n) noexcept
        {
            interpcpy_cc(i,j,n,fine_in,crse,fine_out);
        });
    } else {
        hpmg::ParallelFor(to2D(valid_domain_box(box)), num_comps,
        [=] AMREX_GPU_DEVICE (int i, int j, int n) noexcept
        {
            interpcpy_nd(i,j,n,fine_in,crse,fine_out);
        });
    }
}



#if defined(AMREX_USE_GPU)

#if defined(AMREX_USE_DPCPP)
#define HPMG_SYNCTHREADS item.barrier(sycl::access::fence_space::global_and_local)
#else
#define HPMG_SYNCTHREADS __syncthreads()
#endif

template <int NS, int system_type, typename FGS, typename FRES>
void bottomsolve_gpu (Real dx0, Real dy0, Array4<Real> const* acf,
                      Array4<Real> const* res, Array4<Real> const* cor,
                      Array4<Real> const* rescor, int nlevs, int corner_offset,
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
        int lenx = cor[0].end.x - cor[0].begin.x - 2*corner_offset;
        int leny = cor[0].end.y - cor[0].begin.y - 2*corner_offset;
        int ncells = lenx*leny;
#if defined(AMREX_USE_DPCPP)
        const int icell = item.get_local_linear_id();
#else
        const int icell = threadIdx.x;
#endif
        int j = icell /   lenx;
        int i = icell - j*lenx;
        j += cor[0].begin.y + corner_offset;
        i += cor[0].begin.x + corner_offset;

        for (int ilev = 0; ilev < nlevs-1; ++ilev) {
            if (icell < ncells) {
                if (system_type == 1 || system_type == 2) {
                    cor[ilev](i,j,0,0) = Real(0.);
                    cor[ilev](i,j,0,1) = Real(0.);
                } else {
                    cor[ilev](i,j,0,0) = Real(0.);
                }
            }
            HPMG_SYNCTHREADS;

            for (int is = 0; is < 4; ++is) {
                if (icell < ncells) {
                    if ((i+j+is)%2 == 0) {
                        fgs(i, j,
                            cor[ilev].begin.x, cor[ilev].begin.y,
                            cor[ilev].end.x-1, cor[ilev].end.y-1,
                            cor[ilev], res[ilev], acf[ilev],
                            facx, facy);
                    }
                }
                HPMG_SYNCTHREADS;
            }

            if (icell < ncells) {
                fres(i, j,
                     rescor[ilev],
                     cor[ilev].begin.x, cor[ilev].begin.y,
                     cor[ilev].end.x-1, cor[ilev].end.y-1,
                     cor[ilev],
                     res[ilev],
                     acf[ilev], facx, facy);
            }
            HPMG_SYNCTHREADS;

            lenx = cor[ilev+1].end.x - cor[ilev+1].begin.x - 2*corner_offset;
            leny = cor[ilev+1].end.y - cor[ilev+1].begin.y - 2*corner_offset;
            ncells = lenx*leny;
            if (icell < ncells) {
                j = icell /   lenx;
                i = icell - j*lenx;
                j += cor[ilev+1].begin.y + corner_offset;
                i += cor[ilev+1].begin.x + corner_offset;
                if (corner_offset == 0) {
                    if (system_type == 1 || system_type == 2) {
                        restrict_cc(i,j,0,res[ilev+1],rescor[ilev]);
                        restrict_cc(i,j,1,res[ilev+1],rescor[ilev]);
                    } else {
                        restrict_cc(i,j,0,res[ilev+1],rescor[ilev]);
                    }
                } else {
                    if (system_type == 1 || system_type == 2) {
                        restrict_nd(i,j,0,res[ilev+1],rescor[ilev]);
                        restrict_nd(i,j,1,res[ilev+1],rescor[ilev]);
                    } else {
                        restrict_nd(i,j,0,res[ilev+1],rescor[ilev]);
                    }
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
                if (system_type == 1 || system_type == 2) {
                    cor[ilev](i,j,0,0) = Real(0.);
                    cor[ilev](i,j,0,1) = Real(0.);
                } else {
                    cor[ilev](i,j,0,0) = Real(0.);
                }
            }
            HPMG_SYNCTHREADS;

            for (int is = 0; is < NS; ++is) {
                if (icell < ncells) {
                    if ((i+j+is)%2 == 0) {
                        fgs(i, j,
                            cor[ilev].begin.x, cor[ilev].begin.y,
                            cor[ilev].end.x-1, cor[ilev].end.y-1,
                            cor[ilev], res[ilev], acf[ilev],
                            facx, facy);
                    }
                }
                HPMG_SYNCTHREADS;
            }
        }

        for (int ilev = nlevs-2; ilev >=0; --ilev) {
            lenx = cor[ilev].end.x - cor[ilev].begin.x - 2*corner_offset;
            leny = cor[ilev].end.y - cor[ilev].begin.y - 2*corner_offset;
            ncells = lenx*leny;
            facx *= Real(4.);
            facy *= Real(4.);

            if (icell < ncells) {
                j = icell /   lenx;
                i = icell - j*lenx;
                j += cor[ilev].begin.y + corner_offset;
                i += cor[ilev].begin.x + corner_offset;
                if (corner_offset == 0) {
                    if (system_type == 1 || system_type == 2) {
                        interpadd_cc(i, j, 0, cor[ilev], cor[ilev+1]);
                        interpadd_cc(i, j, 1, cor[ilev], cor[ilev+1]);
                    } else {
                        interpadd_cc(i, j, 0, cor[ilev], cor[ilev+1]);
                    }
                } else {
                    if (system_type == 1 || system_type == 2) {
                        interpadd_nd(i, j, 0, cor[ilev], cor[ilev+1]);
                        interpadd_nd(i, j, 1, cor[ilev], cor[ilev+1]);
                    } else {
                        interpadd_nd(i, j, 0, cor[ilev], cor[ilev+1]);
                    }
                }
            }

            for (int is = 0; is < 4; ++is) {
                HPMG_SYNCTHREADS;
                if (icell < ncells) {
                    if ((i+j+is)%2 == 0) {
                        fgs(i, j,
                            cor[ilev].begin.x, cor[ilev].begin.y,
                            cor[ilev].end.x-1, cor[ilev].end.y-1,
                            cor[ilev], res[ilev], acf[ilev],
                            facx, facy);
                    }
                }
            }
        }
    });
}

#endif // AMREX_USE_GPU

} // namespace {}

MultiGrid::MultiGrid (Real dx, Real dy, Box a_domain, int a_system_type)
    : m_system_type(a_system_type), m_dx(dx), m_dy(dy)
{
    m_num_comps = get_num_comps(m_system_type);
    m_num_comps_acf = get_num_comps_acf(m_system_type);

    IntVect const a_domain_len = a_domain.length();

    AMREX_ALWAYS_ASSERT(a_domain_len[2] == 1 && a_domain.cellCentered() &&
                        a_domain_len[0]%2 == a_domain_len[1]%2);

    IndexType const index_type = (a_domain_len[0]%2 == 0) ?
        IndexType::TheCellType() : IndexType(IntVect(1,1,0));
    m_domain.push_back(amrex::makeSlab(Box{{0,0,0}, a_domain_len-1, index_type}, 2, 0));
    if (!index_type.cellCentered()) {
        m_domain[0].growHi(0,2).growHi(1,2);
        m_boundary_condition_offset = 1.;
        m_boundary_condition_factor = 1.;
    } else {
        m_boundary_condition_offset = 0.5;
        m_boundary_condition_factor = 8./3.;
    }
    IntVect const min_width = index_type.cellCentered() ? IntVect(2,2,1) : IntVect(4,4,1);
    for (int i = 0; i < 30; ++i) {
        if (m_domain.back().coarsenable(IntVect(2,2,1), min_width)) {
            m_domain.push_back(amrex::coarsen(m_domain.back(),IntVect(2,2,1)));
        } else {
            break;
        }
    }
    m_max_level = m_domain.size()-1;
#if defined(AMREX_USE_GPU)
    auto r = std::find_if(std::begin(m_domain), std::end(m_domain),
                          [=] (Box const& b) -> bool
                              { return b.volume() <= n_cell_single*n_cell_single; });
    m_single_block_level_begin = std::distance(std::begin(m_domain), r);
    m_single_block_level_begin = std::max(1, m_single_block_level_begin);
    if (m_single_block_level_begin > m_max_level) {
        m_single_block_level_begin = m_max_level;
        m_use_single_block_kernel = false;
        amrex::Print() << "hpmg: WARNING domain of size "
            << a_domain_len[0] << " " << a_domain_len[1]
            << " cannot be coarsened enough times to be solved efficiently.\n"
            << "hpmg: Size of the final MG level: "
            << m_domain[m_max_level].length(0) << " " << m_domain[m_max_level].length(1) << ".\n"
            << "hpmg: Please consider using a domain size of the form '2^n', '3*2^n', '2^n+1' or '3*n^2+1'.\n";
    }
#else
    m_single_block_level_begin = m_max_level;
#endif
    AMREX_ALWAYS_ASSERT(m_single_block_level_begin > 0);

    m_num_mg_levels = m_max_level+1;
    m_num_single_block_levels = m_num_mg_levels - m_single_block_level_begin;

    if (m_num_single_block_levels > 0) {
        m_h_array4.reserve(nfabvs*m_num_single_block_levels);
    }

    m_acf.reserve(m_num_mg_levels);
    for (int ilev = 0; ilev < m_num_mg_levels; ++ilev) {
        m_acf.emplace_back(m_domain[ilev], m_num_comps_acf);
        if (ilev >= m_single_block_level_begin) {
            m_h_array4.push_back(m_acf[ilev].array());
        }
    }

    m_res.reserve(m_num_mg_levels);
    for (int ilev = 0; ilev < m_num_mg_levels; ++ilev) {
        if (ilev == 0) {
            // rhs is used instead of res[0]
            m_res.emplace_back();
        } else {
            m_res.emplace_back(m_domain[ilev], m_num_comps);
            if (!index_type.cellCentered()) {
                m_res[ilev].template setVal<RunOn::Device>(0);
            }
            if (ilev >= m_single_block_level_begin) {
                m_h_array4.push_back(m_res[ilev].array());
            }
        }
    }

    m_cor.reserve(m_num_mg_levels);
    for (int ilev = 0; ilev < m_num_mg_levels; ++ilev) {
        m_cor.emplace_back(m_domain[ilev], m_num_comps);
        if (!index_type.cellCentered()) {
            m_cor[ilev].template setVal<RunOn::Device>(0);
        }
        if (ilev >= m_single_block_level_begin) {
            m_h_array4.push_back(m_cor[ilev].array());
        }
    }

    m_rescor.reserve(m_num_mg_levels);
    for (int ilev = 0; ilev < m_num_mg_levels; ++ilev) {
        m_rescor.emplace_back(m_domain[ilev], m_num_comps);
        if (!index_type.cellCentered()) {
            m_rescor[ilev].template setVal<RunOn::Device>(0);
        }
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
    AMREX_ALWAYS_ASSERT(m_system_type == 1);

    FArrayBox afab(center_box(a_acf.box(), m_domain.front()), 1, a_acf.dataPtr());

    auto const& array_m_acf = m_acf[0].array();
    auto const& array_a_acf = afab.const_array();
    hpmg::ParallelFor(to2D(m_acf[0].box()),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            array_m_acf(i,j,0) = array_a_acf(i,j,0);
        });

    average_down_acoef();

    solve_doit(a_sol, a_rhs, tol_rel, tol_abs, nummaxiter, verbose);
}

void
MultiGrid::solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                   amrex::Real const acoef_real, amrex::Real const acoef_imag,
                   amrex::Real const tol_rel, amrex::Real const tol_abs,
                   int const nummaxiter, int const verbose)
{
    HIPACE_PROFILE("hpmg::MultiGrid::solve2()");
    AMREX_ALWAYS_ASSERT(m_system_type == 2);

    auto const& array_m_acf = m_acf[0].array();

    hpmg::ParallelFor(to2D(m_acf[0].box()),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            array_m_acf(i,j,0,0) = acoef_real;
            array_m_acf(i,j,0,1) = acoef_imag;
        });

    average_down_acoef();

    solve_doit(sol, rhs, tol_rel, tol_abs, nummaxiter, verbose);
}

void
MultiGrid::solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                   amrex::Real const acoef_real, amrex::FArrayBox const& acoef_imag,
                   amrex::Real const tol_rel, amrex::Real const tol_abs,
                   int const nummaxiter, int const verbose)
{
    HIPACE_PROFILE("hpmg::MultiGrid::solve2()");
    AMREX_ALWAYS_ASSERT(m_system_type == 2);

    auto const& array_m_acf = m_acf[0].array();

    amrex::FArrayBox ifab(center_box(acoef_imag.box(), m_domain.front()), 1, acoef_imag.dataPtr());
    auto const& ai = ifab.const_array();
    hpmg::ParallelFor(to2D(m_acf[0].box()),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            array_m_acf(i,j,0,0) = acoef_real;
            array_m_acf(i,j,0,1) = ai(i,j,0);
        });

    average_down_acoef();

    solve_doit(sol, rhs, tol_rel, tol_abs, nummaxiter, verbose);
}

void
MultiGrid::solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                   amrex::FArrayBox const& acoef_real, amrex::Real const acoef_imag,
                   amrex::Real const tol_rel, amrex::Real const tol_abs,
                   int const nummaxiter, int const verbose)
{
    HIPACE_PROFILE("hpmg::MultiGrid::solve2()");
    AMREX_ALWAYS_ASSERT(m_system_type == 2);

    auto const& array_m_acf = m_acf[0].array();

    amrex::FArrayBox rfab(center_box(acoef_real.box(), m_domain.front()), 1, acoef_real.dataPtr());
    auto const& ar = rfab.const_array();
    hpmg::ParallelFor(to2D(m_acf[0].box()),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            array_m_acf(i,j,0,0) = ar(i,j,0);
            array_m_acf(i,j,0,1) = acoef_imag;
        });

    average_down_acoef();

    solve_doit(sol, rhs, tol_rel, tol_abs, nummaxiter, verbose);
}

void
MultiGrid::solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                   amrex::FArrayBox const& acoef_real, amrex::FArrayBox const& acoef_imag,
                   amrex::Real const tol_rel, amrex::Real const tol_abs,
                   int const nummaxiter, int const verbose)
{
    HIPACE_PROFILE("hpmg::MultiGrid::solve2()");
    AMREX_ALWAYS_ASSERT(m_system_type == 2);

    auto const& array_m_acf = m_acf[0].array();

    amrex::FArrayBox rfab(center_box(acoef_real.box(), m_domain.front()), 1, acoef_real.dataPtr());
    amrex::FArrayBox ifab(center_box(acoef_imag.box(), m_domain.front()), 1, acoef_imag.dataPtr());
    auto const& ar = rfab.const_array();
    auto const& ai = ifab.const_array();
    hpmg::ParallelFor(to2D(m_acf[0].box()),
        [=] AMREX_GPU_DEVICE (int i, int j) noexcept
        {
            array_m_acf(i,j,0,0) = ar(i,j,0);
            array_m_acf(i,j,0,1) = ai(i,j,0);
        });

    average_down_acoef();

    solve_doit(sol, rhs, tol_rel, tol_abs, nummaxiter, verbose);
}

void
MultiGrid::solve3 (FArrayBox& a_sol, FArrayBox const& a_rhs,
                   Real const tol_rel, Real const tol_abs, int const nummaxiter,
                   int const verbose)
{
    HIPACE_PROFILE("hpmg::MultiGrid::solve3()");
    AMREX_ALWAYS_ASSERT(m_system_type == 3);

    solve_doit(a_sol, a_rhs, tol_rel, tol_abs, nummaxiter, verbose);
}


void
MultiGrid::solve_doit (FArrayBox& a_sol, FArrayBox const& a_rhs,
                       Real const tol_rel, Real const tol_abs, int const nummaxiter,
                       int const verbose)
{
    AMREX_ALWAYS_ASSERT(a_sol.nComp() >= m_num_comps && a_rhs.nComp() >= m_num_comps);

    m_sol = FArrayBox(center_box(a_sol.box(), m_domain.front()), m_num_comps, a_sol.dataPtr());
    m_rhs = FArrayBox(center_box(a_rhs.box(), m_domain.front()), m_num_comps, a_rhs.dataPtr());

    // cor = gsrb(gsrb(gsrb(gsrb(sol))))
    // rescor = rhs - L(cor)
    gsrb_4_residual<false, true>(m_system_type, m_domain[0], m_cor[0].array(), m_rhs.const_array(),
        m_acf[0].const_array(), m_rescor[0].array(), m_sol.array(), m_dx, m_dy);

    Real resnorm0, rhsnorm0;
    {
        ReduceOps<ReduceOpMax,ReduceOpMax> reduce_op;
        ReduceData<Real,Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        const auto& array_res = m_rescor[0].const_array();
        const auto& array_rhs = m_rhs.const_array();
        reduce_op.eval(valid_domain_box(m_domain[0]), m_num_comps, reduce_data,
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

            Real const* pres0 = m_rescor[0].dataPtr();
            norminf = Reduce::Max<Real>(m_domain[0].numPts()*m_num_comps,
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

    auto const& sol = m_sol.array();
    auto const& cor = m_cor[0].const_array();
    hpmg::ParallelFor(to2D(valid_domain_box(m_domain[0])), m_num_comps,
    [=] AMREX_GPU_DEVICE (int i, int j, int n) noexcept
    {
        sol(i,j,0,n) = cor(i,j,0,n);
    });
}

void
MultiGrid::vcycle ()
{
#if defined(AMREX_USE_CUDA)
    std::pair<const amrex::Real*, const amrex::Real*> key {
        m_sol.array().dataPtr(),
        m_rhs.const_array().dataPtr()
    };

    if (m_cuda_graph_vcycle.count(key) == 0) {
        cudaStreamBeginCapture(Gpu::gpuStream(), cudaStreamCaptureModeGlobal);
#endif

    for (int ilev = 0; ilev < m_single_block_level_begin; ++ilev) {

        Real fac = static_cast<Real>(1 << ilev);
        Real dx = m_dx * fac;
        Real dy = m_dy * fac;

        if (ilev > 0) {
            // cor and residual on ilev 0 are already calculated before the vcycle is started

            // cor = gsrb(gsrb(gsrb(gsrb(0))))
            // rescor = res - L(cor)
            gsrb_4_residual<true, true>(
                m_system_type, m_domain[ilev], m_cor[ilev].array(), m_res[ilev].const_array(),
                m_acf[ilev].const_array(), m_rescor[ilev].array(), {}, dx, dy);
        }

        // res[ilev+1] = R(rescor[ilev])
        restriction(m_domain[ilev+1], m_res[ilev+1].array(), m_rescor[ilev].const_array(), m_num_comps);
    }

    bottomsolve();

    for (int ilev = m_single_block_level_begin-1; ilev >= 0; --ilev) {

        Real fac = static_cast<Real>(1 << ilev);
        Real dx = m_dx * fac;
        Real dy = m_dy * fac;

        // rescor[ilev] = I(cor[ilev+1]) + cor[ilev]
        interpolation_outofplace(m_domain[ilev], m_cor[ilev].const_array(),
                                 m_cor[ilev+1].const_array(), m_rescor[ilev].array(), m_num_comps);

        if (ilev == 0) {
            // sol = gsrb(gsrb(gsrb(gsrb(rescor))))
            gsrb_4_residual<false, false>(
                    m_system_type, m_domain[ilev], m_sol.array(), m_rhs.const_array(),
                    m_acf[ilev].const_array(), {}, m_rescor[ilev].array(), dx, dy);
        } else {
            // cor = gsrb(gsrb(gsrb(gsrb(rescor))))
            gsrb_4_residual<false, false>(
                    m_system_type, m_domain[ilev], m_cor[ilev].array(), m_res[ilev].const_array(),
                    m_acf[ilev].const_array(), {}, m_rescor[ilev].array(), dx, dy);
        }
    }

    // cor = gsrb(gsrb(gsrb(gsrb(sol))))
    // rescor = rhs - L(cor)
    gsrb_4_residual<false, true>(
        m_system_type, m_domain[0], m_cor[0].array(), m_rhs.const_array(),
        m_acf[0].const_array(), m_rescor[0].array(), m_sol.array(), m_dx, m_dy);

#if defined(AMREX_USE_CUDA)
        cudaStreamEndCapture(Gpu::gpuStream(), &m_cuda_graph_vcycle[key].first);
        cudaGraphInstantiate(&m_cuda_graph_vcycle[key].second,
                            m_cuda_graph_vcycle[key].first, NULL, NULL, 0);
    }
    cudaGraphLaunch(m_cuda_graph_vcycle[key].second, Gpu::gpuStream());
#endif
}

void
MultiGrid::bottomsolve ()
{
    constexpr int nsweeps = 16;
    Real fac = static_cast<Real>(1 << m_single_block_level_begin);
    Real dx0 = m_dx * fac;
    Real dy0 = m_dy * fac;
#if defined(AMREX_USE_GPU)
    if (m_use_single_block_kernel) {
        int nlevs = m_num_single_block_levels;
        int const corner_offset = m_domain[0].cellCentered() ? 0 : 1;

        if (m_system_type == 1) {
            bottomsolve_gpu<nsweeps,1>(dx0, dy0, m_acf_a, m_res_a, m_cor_a, m_rescor_a, nlevs, corner_offset,
                [] AMREX_GPU_DEVICE (int i, int j, int ilo, int jlo, int ihi, int jhi,
                                      Array4<Real> const& phi, Array4<Real> const& rhs,
                                      Array4<Real> const& acf, Real facx, Real facy)
                {
                    Real a = acf(i,j,0);
                    gs1(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), a, facx, facy);
                    gs1(i, j, 1, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,1), a, facx, facy);
                },
                [] AMREX_GPU_DEVICE (int i, int j, Array4<Real> const& res,
                                      int ilo, int jlo, int ihi, int jhi,
                                      Array4<Real> const& phi, Array4<Real> const& rhs,
                                      Array4<Real> const& acf, Real facx, Real facy)
                {
                    Real a = acf(i,j,0);
                    res(i,j,0,0) = residual1(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), a, facx, facy);
                    res(i,j,0,1) = residual1(i, j, 1, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,1), a, facx, facy);
                });
        } else if (m_system_type == 2) {
            bottomsolve_gpu<nsweeps,2>(dx0, dy0, m_acf_a, m_res_a, m_cor_a, m_rescor_a, nlevs, corner_offset,
                [] AMREX_GPU_DEVICE (int i, int j, int ilo, int jlo, int ihi, int jhi,
                                      Array4<Real> const& phi, Array4<Real> const& rhs,
                                      Array4<Real> const& acf, Real facx, Real facy)
                {
                    Real ar = acf(i,j,0,0);
                    Real ai = acf(i,j,0,1);
                    gs2(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), rhs(i,j,0,1), ar, ai, facx, facy);
                },
                [] AMREX_GPU_DEVICE (int i, int j, Array4<Real> const& res,
                                      int ilo, int jlo, int ihi, int jhi,
                                      Array4<Real> const& phi, Array4<Real> const& rhs,
                                      Array4<Real> const& acf, Real facx, Real facy)
                {
                    Real ar = acf(i,j,0,0);
                    Real ai = acf(i,j,0,1);
                    res(i,j,0,0) = residual2r(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), ar, ai, facx, facy);
                    res(i,j,0,1) = residual2i(i, j, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,1), ar, ai, facx, facy);
                });
        } else {
            bottomsolve_gpu<nsweeps,3>(dx0, dy0, m_acf_a, m_res_a, m_cor_a, m_rescor_a, nlevs, corner_offset,
                [] AMREX_GPU_DEVICE (int i, int j, int ilo, int jlo, int ihi, int jhi,
                                      Array4<Real> const& phi, Array4<Real> const& rhs,
                                      Array4<Real> const&, Real facx, Real facy)
                {
                    gs3(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), facx, facy);
                },
                [] AMREX_GPU_DEVICE (int i, int j, Array4<Real> const& res,
                                      int ilo, int jlo, int ihi, int jhi,
                                      Array4<Real> const& phi, Array4<Real> const& rhs,
                                      Array4<Real> const&, Real facx, Real facy)
                {
                    res(i,j,0,0) = residual3(i, j, 0, ilo, jlo, ihi, jhi, phi, rhs(i,j,0,0), facx, facy);
                });
        }
    } else
#endif
    {
        const int ilev = m_single_block_level_begin;
        m_cor[ilev].setVal<amrex::RunOn::Device>(Real(0.));
        // Use numsweeps equal to the box length rounded up to an even number for large boxes
        int numsweeps = std::max(nsweeps, (m_cor[ilev].box().length().max() + 1) / 2 * 2);
        for (int is = 0; is < numsweeps; ++is) {
            gsrb(is, m_domain[ilev], m_cor[ilev].array(),
                m_res[ilev].const_array(), m_acf[ilev].const_array(), dx0, dy0,
                m_system_type);
        }
    }
}

#if defined(AMREX_USE_GPU)
namespace {
    template <typename F>
    void avgdown_acf (Array4<Real> const* acf, int ncomp, int nlevels, F&& f)
    {
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
                        f(i,j,n,acf[ilev],acf[ilev-1]);
                    }
                }
                HPMG_SYNCTHREADS;
            }
        });
    }
}
#endif

void
MultiGrid::average_down_acoef ()
{
#if defined(AMREX_USE_CUDA)
    if (!m_cuda_graph_acf_created) {
    cudaStreamBeginCapture(Gpu::gpuStream(), cudaStreamCaptureModeGlobal);
#endif

    for (int ilev = 1; ilev <= m_single_block_level_begin; ++ilev) {
        auto const& crse = m_acf[ilev].array();
        auto const& fine = m_acf[ilev-1].const_array();
        if (m_domain[ilev].cellCentered()) {
            hpmg::ParallelFor(to2D(m_domain[ilev]), m_num_comps_acf,
            [=] AMREX_GPU_DEVICE (int i, int j, int n) noexcept
            {
                restrict_cc(i,j,n,crse,fine);
            });
        } else {
            hpmg::ParallelFor(to2D(valid_domain_box(m_domain[ilev])), m_num_comps_acf,
            [=] AMREX_GPU_DEVICE (int i, int j, int n) noexcept
            {
                restrict_nd(i,j,n,crse,fine);
            });
        }
    }

#if defined(AMREX_USE_GPU)
    if (m_num_single_block_levels > 1) {
        if (m_domain[0].cellCentered()) {
            avgdown_acf(m_acf_a, m_num_comps_acf, m_num_single_block_levels,
                        [] AMREX_GPU_DEVICE (int i, int j, int n, Array4<Real> const& crse,
                                             Array4<Real> const& fine) noexcept
                        {
                            restrict_cc(i,j,n,crse,fine);
                        });
        } else {
            avgdown_acf(m_acf_a, m_num_comps_acf, m_num_single_block_levels,
                        [] AMREX_GPU_DEVICE (int i, int j, int n, Array4<Real> const& crse,
                                             Array4<Real> const& fine) noexcept
                        {
                            if (i == crse.begin.x ||
                                j == crse.begin.y ||
                                i == crse.end.x-1 ||
                                j == crse.end.y-1) {
                                crse(i,j,0,n) = Real(0.);
                            } else {
                                restrict_nd(i,j,n,crse,fine);
                            }
                        });
        }
    }
#endif

#if defined(AMREX_USE_CUDA)
    cudaStreamEndCapture(Gpu::gpuStream(), &m_cuda_graph_acf);
    cudaGraphInstantiate(&m_cuda_graph_exe_acf, m_cuda_graph_acf, NULL, NULL, 0);
    m_cuda_graph_acf_created = true;
    }
    cudaGraphLaunch(m_cuda_graph_exe_acf, Gpu::gpuStream());
#endif
}

MultiGrid::~MultiGrid ()
{
#if defined(AMREX_USE_CUDA)
    if (m_cuda_graph_acf_created) {
        cudaGraphDestroy(m_cuda_graph_acf);
        cudaGraphExecDestroy(m_cuda_graph_exe_acf);
    }
    for (auto& graph : m_cuda_graph_vcycle) {
        cudaGraphDestroy(graph.second.first);
        cudaGraphExecDestroy(graph.second.second);
    }
#endif
}

}
