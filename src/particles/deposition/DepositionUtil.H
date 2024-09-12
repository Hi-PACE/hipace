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

template<int stencil_x, int stencil_y,
         class F1, class F2, class F3,
         std::size_t max_depos, std::size_t max_cache,
         class PTD>
void
SharedMemoryDeposition (int num_particles,
                        F1&& is_valid, [[maybe_unused]] F2&& get_start_cell, F3&& do_deposit,
                        Array3<amrex::Real> field, [[maybe_unused]] amrex::Box box, const PTD& ptd,
                        std::array<int, max_cache> idx_cache,
                        std::array<int, max_depos> idx_depos) {

    if (Hipace::m_do_shared_depos) {
#ifdef AMREX_USE_GPU
        constexpr int threads_per_tile = 256;
        constexpr int tile_x = 16;
        constexpr int tile_y = 16;
        constexpr int tile_s_x = tile_x + stencil_x - 1;
        constexpr int tile_s_y = tile_y + stencil_y - 1;

        const int lo_x = box.smallEnd(0);
        const int lo_y = box.smallEnd(1);
        const int hi_x = box.bigEnd(0);
        const int hi_y = box.bigEnd(1);
        const int ntile_x = (box.length(0) + tile_x - 1) / tile_x;
        const int ntile_y = (box.length(1) + tile_y - 1) / tile_y;
        constexpr int ll_guard = std::numeric_limits<int>::max();
        amrex::Gpu::DeviceVector<int> ll_start(ntile_x * ntile_y * threads_per_tile, ll_guard);
        amrex::Gpu::DeviceVector<int> ll_count(ntile_x * ntile_y * 64, 0);
        amrex::Gpu::DeviceVector<int> ll_next(num_particles);
        int * const p_ll_start = ll_start.dataPtr();
        int * const p_ll_count = ll_count.dataPtr();
        int * const p_ll_next = ll_next.dataPtr();

        {
        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (int ip) {
                //ip = num_particles - ip - 1;

                if (is_valid(ip, ptd)) {
                    auto [cell_x, cell_y] = get_start_cell(ip, ptd);

                    const int tile_id_x = (cell_x - lo_x) / tile_x;
                    const int tile_id_y = (cell_y - lo_y) / tile_y;
                    const int idx = (tile_id_x + tile_id_y * ntile_x);

                    //const int count = amrex::Gpu::Atomic::Add(p_ll_count + idx, 1);

                    //p_ll_next[ip] = amrex::Gpu::Atomic::Exch(
                    //    p_ll_start + idx*threads_per_tile + count % threads_per_tile, ip);

                    const int loc_id_x = (cell_x - lo_x - tile_id_x * tile_x);
                    const int loc_id_y = (cell_y - lo_y - tile_id_y * tile_y);
                    const int loc_id = loc_id_x + loc_id_y * tile_x;

                    const int count = amrex::Gpu::Atomic::Add(
                        p_ll_count + idx*64 + (loc_id % 64), 1);

                    p_ll_next[ip] = amrex::Gpu::Atomic::Exch(
                        p_ll_start + idx*threads_per_tile + (loc_id % 64) + (count % 4) * 64 , ip);

                    //p_ll_next[ip] = amrex::Gpu::Atomic::Exch(
                    //    p_ll_start + idx*threads_per_tile + loc_id, ip);
                }
            });

        }


        /*{

            amrex::launch<threads_per_tile>(ntile_x * ntile_y, amrex::Gpu::gpuStream(),
                [=] AMREX_GPU_DEVICE () {
                    const int tile_id = blockIdx.x;
                    const int thread_id = threadIdx.x;

                    __shared__ int counts[threads_per_tile];
                    __shared__ int old_ll_start[threads_per_tile];
                    __shared__ int new_ll_start[threads_per_tile];


                    counts[thread_id] = 0;
                    old_ll_start[thread_id] = p_ll_start[tile_id * threads_per_tile + thread_id];
                    new_ll_start[thread_id] = ll_guard;

                    int current_idx = old_ll_start[thread_id];

                    while (current_idx != ll_guard) {

                        counts[thread_id] += 1;

                        current_idx = p_ll_next[current_idx];
                    }

                    __syncthreads();

                    if (thread_id == 0) {

                        int total_count = 0;

                        for (int i=0; i<threads_per_tile; ++i) {
                            total_count += counts[i];
                        }

                        int j = 0;

                        while (total_count > 0) {

                            for (int i=0; i<threads_per_tile; ++i) {

                                if (old_ll_start[i] != ll_guard) {

                                    int tmp = new_ll_start[j%threads_per_tile];
                                    new_ll_start[j%threads_per_tile] = old_ll_start[i];
                                    old_ll_start[i] = p_ll_next[new_ll_start[j%threads_per_tile]];
                                    p_ll_next[new_ll_start[j%threads_per_tile]] = tmp;

                                    --total_count;
                                    ++j;
                                }
                            }
                        }
                    }

                    __syncthreads();

                    //const int mod = 16;

                    //int perm = (thread_id + (thread_id%mod)*(threads_per_tile/mod)) % threads_per_tile;

                    p_ll_start[tile_id * threads_per_tile + thread_id] = new_ll_start[thread_id];


                });
        }*/


        /*{

            amrex::launch<threads_per_tile>(ntile_x * ntile_y, amrex::Gpu::gpuStream(),
                [=] AMREX_GPU_DEVICE () {
                    const int tile_id = blockIdx.x;
                    const int thread_id = threadIdx.x;

                    __shared__ int counts[threads_per_tile];

                    counts[thread_id] = 0;

                    int current_idx = p_ll_start[tile_id * threads_per_tile + thread_id];

                    while (current_idx != ll_guard) {

                        counts[thread_id] += 1;

                        current_idx = p_ll_next[current_idx];
                    }

                    __syncthreads();


                    if (thread_id == 0) {

                        int total = 0;

                        for (int i=0; i<threads_per_tile; ++i) {
                            total += counts[i];
                        }

                        int extra_list = ll_guard;

                        for (int i=0; i<threads_per_tile; ++i) {
                            int target = (total + i) / threads_per_tile;

                            while (counts[i] > target) {
                                int prev_extra_list = extra_list;
                                extra_list = p_ll_start[tile_id * threads_per_tile + i];
                                p_ll_start[tile_id * threads_per_tile + i] = p_ll_next[extra_list];
                                p_ll_next[extra_list] = prev_extra_list;

                                counts[i] -= 1;
                            }
                        }

                        for (int i=0; i<threads_per_tile; ++i) {
                            int target = (total + i) / threads_per_tile;

                            while (counts[i] < target) {
                                int prev_ll_start = p_ll_start[tile_id * threads_per_tile + i];
                                p_ll_start[tile_id * threads_per_tile + i] = extra_list;
                                extra_list = p_ll_next[p_ll_start[tile_id * threads_per_tile + i]];
                                p_ll_next[p_ll_start[tile_id * threads_per_tile + i]] = prev_ll_start;

                                counts[i] += 1;
                            }
                        }

                    }

                });
        }*/

        /*{
            amrex::Gpu::DeviceVector<int> ll_start_2(ll_start.size());
            int * const p_ll_start_2 = ll_start_2.dataPtr();
            amrex::ParallelFor(ll_start.size(),
                [=] AMREX_GPU_DEVICE (int ip) {
                    p_ll_start_2[ip] = p_ll_start[ip];
                });

            std::cout << "depos";
            int num_par = 0;
            do {
                num_par = amrex::Reduce::Sum(ll_start.size(), [=]AMREX_GPU_DEVICE(int ip){
                    int idx = p_ll_start_2[ip];
                    if (idx != ll_guard) {
                        p_ll_start_2[ip] = p_ll_next[idx];
                        return 1;
                    } else {
                        return 0;
                    }
                }, 0);
                std::cout << " " << num_par;
            } while (num_par > 0);
            std::cout << std::endl;
        }*/

        amrex::Math::FastDivmodU64 num_tiles_divmod {static_cast<std::uint64_t>(ntile_x)};


        amrex::launch<threads_per_tile>(ntile_x * ntile_y, amrex::Gpu::gpuStream(),
            [=] AMREX_GPU_DEVICE () {

                __shared__ amrex::Real shared_ptr[tile_s_x * tile_s_y * (max_cache + max_depos)];

                const int tile_id = blockIdx.x;

                std::uint64_t remainder = 0;
                const int tile_id_y = num_tiles_divmod.divmod(remainder, tile_id);
                const int tile_id_x = remainder;

                const int tile_begin_x = lo_x + tile_id_x * tile_x;
                const int tile_begin_y = lo_y + tile_id_y * tile_y;

                const int tile_end_x = std::min(tile_begin_x + tile_s_x, hi_x + 1);
                const int tile_end_y = std::min(tile_begin_y + tile_s_y, hi_y + 1);

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
                            if (idx_cache[n] != -1) {
                                shared_arr(sx, sy, n) = field(sx, sy, idx_cache[n]);
                            }
                        }
                        for (int n=0; n != max_depos; ++n) {
                            if (idx_depos[n] != -1) {
                                shared_arr(sx, sy, n+max_cache) = 0;
                            }
                        }
                    }
                }

                std::array<int, max_cache> loc_idx_cache;
                std::array<int, max_depos> loc_idx_depos;

                for (int n=0; n != max_cache; ++n) {
                    loc_idx_cache[n] = idx_cache[n] == -1 ? -1 : n;
                }

                for (int n=0; n != max_depos; ++n) {
                    loc_idx_depos[n] = idx_depos[n] == -1 ? -1 : n+int(max_cache);
                }

                __syncthreads();

                int current_idx = p_ll_start[tile_id * threads_per_tile + threadIdx.x];

                while (current_idx != ll_guard) {

                    do_deposit(current_idx, ptd, shared_arr, loc_idx_cache, loc_idx_depos);

                    current_idx = p_ll_next[current_idx];
                }

                __syncthreads();

                for (int s = threadIdx.x; s < tile_s_x * tile_s_y; s+=threads_per_tile) {
                    int sy = s / tile_s_x;
                    int sx = s - sy * tile_s_x;
                    sx += tile_begin_x;
                    sy += tile_begin_y;
                    if (sx <= hi_x && sy <= hi_y) {
                        for (int n=0; n != max_depos; ++n) {
                            if (idx_depos[n] != -1) {
                                amrex::Gpu::Atomic::Add(
                                    field.ptr(sx, sy, idx_depos[n]), shared_arr(sx, sy, n+max_cache));
                            }
                        }
                    }
                }


            }
        );

        amrex::Gpu::streamSynchronize();
#endif
    } else {

        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (int ip) {
                if (is_valid(ip, ptd)) {
                    do_deposit(ip, ptd, field, idx_cache, idx_depos);
                }
            });

    }
}

#endif
