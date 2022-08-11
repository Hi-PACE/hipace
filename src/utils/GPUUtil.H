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
        // slice is only one cell thick
        AMREX_ALWAYS_ASSERT(rhs.begin.z + 1 == rhs.end.z);
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
        // slice is only one cell thick
        AMREX_ALWAYS_ASSERT(rhs.begin.z + 1 == rhs.end.z);
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

// reduce size of lambda capture for this object from sizeof(T) to 1
template<bool has_obj, class T>
struct LambdaOptional {
    T data;
    LambdaOptional (const T& in) :data{in} {}
    LambdaOptional (T&& in) :data{in} {}
    LambdaOptional ()=default;
    LambdaOptional& operator= (const T& in) {data=in; return *this;}
    LambdaOptional& operator= (T&& in) {data=in; return *this;}
    AMREX_GPU_HOST_DEVICE T& operator() () noexcept {return data;}
    AMREX_GPU_HOST_DEVICE const T& operator() () const noexcept {return data;}
};

template<class T>
struct LambdaOptional<false, T> {
    LambdaOptional (const T&) {}
    LambdaOptional (T&&) {}
    LambdaOptional ()=default;
    LambdaOptional& operator= (const T&) {return *this;}
    LambdaOptional& operator= (T&&) {return *this;}
};

#endif
