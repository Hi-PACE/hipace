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

    Array2 (const amrex::Array4<T>& rhs) noexcept
        : p(rhs.p),
          jstride(rhs.jstride),
          start(-rhs.begin.x - rhs.begin.y * jstride)
    {
        // slice is only one cell thick
        AMREX_ALWAYS_ASSERT(rhs.begin.z + 1 == rhs.end.z);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T& operator() (int i, int j) const noexcept {
        return p[i + j*jstride + start];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T* ptr (int i, int j) const noexcept {
        return p + (i + j*jstride + start);
    }
};

template<class T>
struct Array3 {
    T* AMREX_RESTRICT p;
    amrex::Long jstride = 0;
    amrex::Long nstride = 0;
    amrex::Long start = 0;

    Array3 (const amrex::Array4<T>& rhs) noexcept
        : p(rhs.p),
          jstride(rhs.jstride),
          nstride(rhs.nstride),
          start(-rhs.begin.x - rhs.begin.y * jstride)
    {
        // slice is only one cell thick
        AMREX_ALWAYS_ASSERT(rhs.begin.z + 1 == rhs.end.z);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T& operator() (int i, int j, int n) const noexcept {
        return p[(start + i + j*jstride) + n*nstride];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T* ptr (int i, int j, int n) const noexcept {
        return p + ((start + i + j*jstride) + n*nstride);
    }
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
