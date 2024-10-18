/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_DEPOSITIONUTIL_H_
#define HIPACE_DEPOSITIONUTIL_H_

#include "Hipace.H"
#include "utils/GPUUtil.H"

#include "AMReX_GpuLaunch.H"

/** Deposit the current / charge of particles onto fields using one of the following methods:
 * GPU: shared memory deposition
 * CPU: 4 color tiling
 * All: simple loop over particles
 *
 * \tparam stencil_x max size in x of the stencil that particles deposit
 * \tparam stencil_y max size in y of the stencil that particles deposit
 * \tparam dynamic_comps
 *             if some components in idx_cache and idx_depos can be disabled by setting them to -1
 * \param[in] num_particles number of particles to deposit
 * \param[in] is_valid functor (int ip, PTD ptd) -> bool
 *            to get if a particle is valid and should deposit
 * \param[in] get_start_cell functor (int ip, PTD ptd) -> IntVectND<2>
 *            to get the lowest cell index that the particle deposits into
 * \param[in] do_deposit
 *            functor (int ip, PTD ptd, Array3 field, array idx_cache, array idx_depos) -> void
 *            to deposit the charge / current of one particle into field using idx_cache, idx_depos
 * \param[in] field field to read from and deposit into
 * \param[in] box box of the field
 * \param[in] ptd ParticleTileData of the particles
 * \param[in] idx_cache indexes of the field components to cache
 * \param[in] idx_depos indexes of the field components to deposit
 */
template<int stencil_x, int stencil_y, bool dynamic_comps,
         class F1, class F2, class F3,
         unsigned int max_depos, unsigned int max_cache,
         class PTD>
AMREX_FLATTEN
void
SharedMemoryDeposition (int num_particles,
                        F1&& is_valid, F2&& get_start_cell, F3&& do_deposit,
                        Array3<amrex::Real> field, amrex::Box box, const PTD& ptd,
                        amrex::GpuArray<int, max_cache> idx_cache,
                        amrex::GpuArray<int, max_depos> idx_depos) {
#ifdef AMREX_USE_GPU
    if (Hipace::m_do_shared_depos) {
        constexpr int threads_per_tile = 256;
        constexpr int tile_x = 16;
        constexpr int tile_y = 16;
        constexpr int tile_s_x = tile_x + stencil_x - 1;
        constexpr int tile_s_y = tile_y + stencil_y - 1;
        constexpr int combine_threads = 4;
        constexpr int combine_stride = threads_per_tile / combine_threads;

        static_assert(tile_x * tile_y == threads_per_tile);
        static_assert(threads_per_tile % combine_threads == 0);

        const int lo_x = box.smallEnd(0);
        const int lo_y = box.smallEnd(1);
        const int hi_x = box.bigEnd(0);
        const int hi_y = box.bigEnd(1);
        const int ntile_x = (box.length(0) + tile_x - 1) / tile_x;
        const int ntile_y = (box.length(1) + tile_y - 1) / tile_y;
        constexpr int ll_guard = std::numeric_limits<int>::max();
        amrex::Gpu::DeviceVector<int> ll_start(ntile_x * ntile_y * threads_per_tile, ll_guard);
        amrex::Gpu::DeviceVector<int> ll_count(ntile_x * ntile_y * combine_stride, 0);
        amrex::Gpu::DeviceVector<int> ll_next(num_particles);
        int * const p_ll_start = ll_start.dataPtr();
        int * const p_ll_count = ll_count.dataPtr();
        int * const p_ll_next = ll_next.dataPtr();

        // build linked lists to assign particles to tiles and cells within those tiles
        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (int ip) {

                if (is_valid(ip, ptd)) {
                    auto [cell_x, cell_y] = get_start_cell(ip, ptd);

                    // compute tile id
                    const int tile_id_x = (cell_x - lo_x) / tile_x;
                    const int tile_id_y = (cell_y - lo_y) / tile_y;
                    const int idx = (tile_id_x + tile_id_y * ntile_x);

                    // compute cell id inside the tile
                    const int loc_id_x = (cell_x - lo_x - tile_id_x * tile_x);
                    const int loc_id_y = (cell_y - lo_y - tile_id_y * tile_y);
                    const int loc_id = loc_id_x + loc_id_y * tile_x;

                    // combine_threads cells share the same count.
                    // The count modulo combine_threads is used to
                    // load balance particles between these cells.
                    const int count = amrex::Gpu::Atomic::Add(
                        p_ll_count + idx*combine_stride + (loc_id % combine_stride), 1);

                    // prepend particle id to the inked list
                    p_ll_next[ip] = amrex::Gpu::Atomic::Exch(
                        p_ll_start + idx*threads_per_tile + (loc_id % combine_stride)
                                   + (count % combine_threads) * combine_stride , ip);
                }
            });

        amrex::Math::FastDivmodU64 num_tiles_divmod {static_cast<std::uint64_t>(ntile_x)};

        // launch shared memory kernel to deposit the charge / current
        // use one block per tile and one thread per cell
        amrex::launch<threads_per_tile>(ntile_x * ntile_y, amrex::Gpu::gpuStream(),
            [=] AMREX_GPU_DEVICE () {
                // allocate static shared memory
                __shared__ amrex::Real shared_ptr[tile_s_x * tile_s_y * (max_cache + max_depos)];

                const int tile_id = blockIdx.x;

                std::uint64_t remainder = 0;
                const int tile_id_y = num_tiles_divmod.divmod(remainder, tile_id);
                const int tile_id_x = remainder;

                const int tile_begin_x = lo_x + tile_id_x * tile_x;
                const int tile_begin_y = lo_y + tile_id_y * tile_y;

                const int tile_end_x = std::min(tile_begin_x + tile_s_x, hi_x + 1);
                const int tile_end_y = std::min(tile_begin_y + tile_s_y, hi_y + 1);

                // make Array3 reference shared memory tile
                Array3<amrex::Real> shared_arr{{
                    shared_ptr,
                    {tile_begin_x, tile_begin_y, 0},
                    {tile_end_x, tile_end_y, 1},
                    max_cache + max_depos
                }};

                for (int s = threadIdx.x; s < tile_s_x * tile_s_y; s+=threads_per_tile) {
                    int sy = s / tile_s_x;
                    int sx = s - sy * tile_s_x;
                    sx += tile_begin_x;
                    sy += tile_begin_y;
                    if (sx <= hi_x && sy <= hi_y) {
                        for (int n=0; n != max_cache; ++n) {
                            if (!dynamic_comps || idx_cache[n] != -1) {
                                // load idx_cache components into shared memory
                                shared_arr(sx, sy, n) = field(sx, sy, idx_cache[n]);
                            }
                        }
                        for (int n=0; n != max_depos; ++n) {
                            if (!dynamic_comps || idx_depos[n] != -1) {
                                // set idx_depos components to zero
                                shared_arr(sx, sy, n+max_cache) = 0;
                            }
                        }
                    }
                }

                // calculate the local field components of the shared memory
                amrex::GpuArray<int, max_cache> loc_idx_cache;
                amrex::GpuArray<int, max_depos> loc_idx_depos;

                for (int n=0; n != max_cache; ++n) {
                    loc_idx_cache[n] = (dynamic_comps && idx_cache[n]==-1) ? -1 : n;
                }

                for (int n=0; n != max_depos; ++n) {
                    loc_idx_depos[n] = (dynamic_comps && idx_depos[n]==-1) ? -1 : n+int(max_cache);
                }

                __syncthreads();

                // deposit the charge / current of all particles in the linked list
                int current_idx = p_ll_start[tile_id * threads_per_tile + threadIdx.x];
                while (current_idx != ll_guard) {
                    do_deposit(current_idx, ptd, shared_arr, loc_idx_cache, loc_idx_depos);
                    current_idx = p_ll_next[current_idx];
                }

                __syncthreads();

                // add to local charge / current to the global one
                for (int s = threadIdx.x; s < tile_s_x * tile_s_y; s+=threads_per_tile) {
                    int sy = s / tile_s_x;
                    int sx = s - sy * tile_s_x;
                    sx += tile_begin_x;
                    sy += tile_begin_y;
                    if (sx <= hi_x && sy <= hi_y) {
                        for (int n=0; n != max_depos; ++n) {
                            if (!dynamic_comps || idx_depos[n] != -1) {
                                amrex::Gpu::Atomic::Add(
                                    field.ptr(sx, sy, idx_depos[n]), shared_arr(sx, sy, n+max_cache));
                            }
                        }
                    }
                }


            }
        );

        // streamSynchronize to safely deallocate the particle linked lists
        amrex::Gpu::streamSynchronize();
    }
#else
    if (Hipace::m_do_tiling) {
        const int tile_x = Hipace::m_tile_size;
        const int tile_y = Hipace::m_tile_size;
        AMREX_ALWAYS_ASSERT(tile_x >= stencil_x && tile_y >= stencil_y);

        const int lo_x = box.smallEnd(0);
        const int lo_y = box.smallEnd(1);
        const int ntile_x = (box.length(0) + tile_x - 1) / tile_x;
        const int ntile_y = (box.length(1) + tile_y - 1) / tile_y;
        amrex::DenseBins<PTD> bins;

        // bin particles by the tile that they deposit into
        bins.build(num_particles, ptd, ntile_x * ntile_y + 1,
            [=] (auto loc_ptd, int ip) {
                if (is_valid(ip, loc_ptd)) {
                    auto [cell_x, cell_y] = get_start_cell(ip, loc_ptd);

                    const int tile_id_x = (cell_x - lo_x) / tile_x;
                    const int tile_id_y = (cell_y - lo_y) / tile_y;
                    return (tile_id_x * ntile_y + tile_id_y);
                } else {
                    return ntile_x * ntile_y;
                }
            });

        int const * const a_indices = bins.permutationPtr();
        int const * const a_offsets = bins.offsetsPtr();

        // 4 color loop over tiles to avoid race conditions between OMP threads
        for (int tile_perm_x = 0; tile_perm_x < 2; ++tile_perm_x) {
            for (int tile_perm_y = 0; tile_perm_y < 2; ++tile_perm_y) {
#ifdef AMREX_USE_OMP
#pragma omp parallel for collapse(2)
#endif
                for (int itile_x = tile_perm_x; itile_x < ntile_x; itile_x += 2) {
                    for (int itile_y = tile_perm_y; itile_y < ntile_y; itile_y += 2) {

                        const int tile_id = (itile_x * ntile_y + itile_y);

#ifdef AMREX_USE_OMP
#pragma omp simd
#endif
                        // deposit charge / current of all particles in this tile
                        for (int ip = a_offsets[tile_id]; ip < a_offsets[tile_id+1]; ++ip) {
                            do_deposit(a_indices[ip], ptd, field, idx_cache, idx_depos);
                        }
                    }
                }
            }
        }
    }
#endif
    else {
        // simple loop over particles, on CPU this only uses one thread
        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (int ip) {
                if (is_valid(ip, ptd)) {
                    do_deposit(ip, ptd, field, idx_cache, idx_depos);
                }
            });
    }
}

#endif
