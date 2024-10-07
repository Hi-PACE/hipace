/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_OMPUTIL_H_
#define HIPACE_OMPUTIL_H_

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>

#include <array>

namespace omp {

#if defined(AMREX_USE_GPU) || !defined(AMREX_USE_OMP)
using amrex::ParallelFor;
#else

// When compiling for CPUs, amrex::ParallelFor only uses a single OMP thread.
// Some ParallelFor loops in hipace are outside MFIter loops, so to use multiple threads
// we need to define our own PrallelFor that uses OMP when compiling for CPU.
// When compiling for GPU the normal amrex::ParallelFor is used.

// 1D ParallelFor
template <typename T, typename F>
void ParallelFor (T n, F&& f) noexcept
{
#pragma omp parallel for simd
    for (T i = 0; i < n; ++i) {
        f(i);
    }
}

// 1D ParallelFor with CTO
template <typename T, class F, typename... CTOs>
void ParallelFor (amrex::TypeList<CTOs...> ctos,
                  std::array<int,sizeof...(CTOs)> const& runtime_options,
                  T N, F&& f)
{
    amrex::AnyCTO(ctos, runtime_options,
        [&](auto cto_func){
            ParallelFor(N, cto_func);
        },
        std::forward<F>(f)
    );
}

#endif
}

#endif
