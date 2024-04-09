/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_GPUUTIL_H_
#define HIPACE_GPUUTIL_H_

#include <AMReX_Array4.H>
#include <AMReX_MFIter.H>

template<class T>
struct Array2 {
    T* AMREX_RESTRICT p;
    amrex::Long jstride = 0;
    amrex::Long start = 0;

#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
    amrex::Dim3 begin{1,1,1};
    amrex::Dim3 end{0,0,0};  // end is hi + 1
    int ncomp=0;
#endif

    Array2 (const amrex::Array4<T>& rhs) noexcept
        : p(rhs.p),
          jstride(rhs.jstride),
          start(-rhs.begin.x - rhs.begin.y * rhs.jstride)
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
          ,begin(rhs.begin),
          end(rhs.end),
          ncomp(rhs.ncomp)
#endif
    {
        // slice is only one cell thick if allocated
        AMREX_ALWAYS_ASSERT(!rhs.p || rhs.begin.z + 1 == rhs.end.z);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T& operator() (int i, int j) const noexcept {
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
        index_assert(i,j,begin.z,0);
#endif
        return p[i + j*jstride + start];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T* ptr (int i, int j) const noexcept {
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
        index_assert(i,j,begin.z,0);
#endif
        return p + (i + j*jstride + start);
    }

#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
    AMREX_GPU_HOST_DEVICE inline
    void index_assert (int i, int j, int k, int n) const
    {
        if (i<begin.x || i>=end.x || j<begin.y || j>=end.y || k<begin.z || k>=end.z
            || n < 0 || n >= ncomp) {
#if AMREX_DEVICE_COMPILE
            AMREX_DEVICE_PRINTF(" (%d,%d,%d,%d) is out of bound (%d:%d,%d:%d,%d:%d,0:%d)\n",
                                i, j, k, n, begin.x, end.x-1, begin.y, end.y-1,
                                begin.z, end.z-1, ncomp-1);
            amrex::Abort();
#else
            std::stringstream ss;
            ss << " (" << i << "," << j << "," << k << "," <<  n
               << ") is out of bound ("
               << begin.x << ":" << end.x-1 << ","
               << begin.y << ":" << end.y-1 << ","
               << begin.z << ":" << end.z-1 << ","
               << "0:" << ncomp-1 << ")";
            amrex::Abort(ss.str());
#endif
        }
    }
#endif
};

template<class T> inline
Array2<T> to_array2 (amrex::Array4<T>&& in) {
    return Array2<T>{in};
}

template<class T> inline
Array2<T> to_array2 (const amrex::Array4<T>& in) {
    return Array2<T>{in};
}

template<class T> inline
T to_array2 (T&& in) {
    return in;
}

template<class T>
struct Array3 {
    T* AMREX_RESTRICT p;
    amrex::Long jstride = 0;
    amrex::Long nstride = 0;
    amrex::Long start = 0;

#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
    amrex::Dim3 begin{1,1,1};
    amrex::Dim3 end{0,0,0};  // end is hi + 1
    int ncomp=0;
#endif

    template <class U=T, std::enable_if_t<std::is_const_v<U>,int> = 0>
    AMREX_GPU_HOST_DEVICE
    Array3 (const Array3<std::remove_const_t<T>>& rhs) noexcept
        : p(rhs.p),
          jstride(rhs.jstride),
          nstride(rhs.nstride),
          start(rhs.start)
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
          ,begin(rhs.begin),
          end(rhs.end),
          ncomp(rhs.ncomp)
#endif
    {}

    Array3 (const amrex::Array4<T>& rhs) noexcept
        : p(rhs.p),
          jstride(rhs.jstride),
          nstride(rhs.nstride),
          start(-rhs.begin.x - rhs.begin.y * rhs.jstride)
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
          ,begin(rhs.begin),
          end(rhs.end),
          ncomp(rhs.ncomp)
#endif
    {
        // slice is only one cell thick if allocated
        AMREX_ALWAYS_ASSERT(!rhs.p || rhs.begin.z + 1 == rhs.end.z);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T& operator() (int i, int j, int n) const noexcept {
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
        index_assert(i,j,begin.z,n);
#endif
        return p[(start + i + j*jstride) + n*nstride];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T* ptr (int i, int j, int n) const noexcept {
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
        index_assert(i,j,begin.z,n);
#endif
        return p + ((start + i + j*jstride) + n*nstride);
    }


#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
    AMREX_GPU_HOST_DEVICE inline
    void index_assert (int i, int j, int k, int n) const
    {
        if (i<begin.x || i>=end.x || j<begin.y || j>=end.y || k<begin.z || k>=end.z
            || n < 0 || n >= ncomp) {
#if AMREX_DEVICE_COMPILE
            AMREX_DEVICE_PRINTF(" (%d,%d,%d,%d) is out of bound (%d:%d,%d:%d,%d:%d,0:%d)\n",
                                i, j, k, n, begin.x, end.x-1, begin.y, end.y-1,
                                begin.z, end.z-1, ncomp-1);
            amrex::Abort();
#else
            std::stringstream ss;
            ss << " (" << i << "," << j << "," << k << "," <<  n
               << ") is out of bound ("
               << begin.x << ":" << end.x-1 << ","
               << begin.y << ":" << end.y-1 << ","
               << begin.z << ":" << end.z-1 << ","
               << "0:" << ncomp-1 << ")";
            amrex::Abort(ss.str());
#endif
        }
    }
#endif
};

template<class T> inline
Array3<T> to_array3 (amrex::Array4<T>&& in) {
    return Array3<T>{in};
}

template<class T> inline
Array3<T> to_array3 (const amrex::Array4<T>& in) {
    return Array3<T>{in};
}

template<class T> inline
T to_array3 (T&& in) {
    return in;
}

inline amrex::MFItInfo DfltMfi; // Default MFIter
inline amrex::MFItInfo DfltMfiTlng; // Default MFIter with Tiling

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real abssq(amrex::Real r, amrex::Real i)
{
    return r*r + i*i;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void AtomicAdd(amrex::Real* ptr, amrex::Real val, [[maybe_unused]] bool do_omp_atomic=true) {
#ifdef AMREX_USE_GPU
    amrex::Gpu::Atomic::Add(ptr, val);
#elif defined(AMREX_USE_OMP)
    if (do_omp_atomic) {
#pragma omp atomic
        *ptr += val;
    } else {
        *ptr += val;
    }
#else
    *ptr += val;
#endif
}

#endif
