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
using Array2 = amrex::ArrayND<T, 2>;

template<class T>
using Array3 = amrex::ArrayND<T, 3>;

template<class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array2<T> to2D (const amrex::Array4<T>& arr) {
    // slice is only one cell thick if allocated
    AMREX_IF_ON_HOST((AMREX_ALWAYS_ASSERT(!arr.p || arr.begin[2] + 1 == arr.end[2]);))
    return Array2<T>(arr.p, {arr.begin[0], arr.begin[1]}, {arr.end[0], arr.end[1]});
}

template<class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array3<T> to3D (const amrex::Array4<T>& arr) {
    // slice is only one cell thick if allocated
    AMREX_IF_ON_HOST((AMREX_ALWAYS_ASSERT(!arr.p || arr.begin[2] + 1 == arr.end[2]);))
    return Array3<T>(arr.p, {arr.begin[0], arr.begin[1], arr.begin[3]},
                            {arr.end[0], arr.end[1], arr.end[3]});
}

template<class T> inline
Array2<T> to_array2 (amrex::Array4<T>&& in) {
    return to2D(in);
}

template<class T> inline
Array2<T> to_array2 (const amrex::Array4<T>& in) {
    return to2D(in);
}

template<class T> inline
T to_array2 (T&& in) {
    return in;
}

template<class T> inline
Array3<T> to_array3 (amrex::Array4<T>&& in) {
    return to3D(in);
}

template<class T> inline
Array3<T> to_array3 (const amrex::Array4<T>& in) {
    return to3D(in);
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
amrex::BoxND<2> to2D (const amrex::BoxND<3>& box) {
    AMREX_IF_ON_HOST((AMREX_ALWAYS_ASSERT(box.length(2) == 1);))
    return amrex::BoxShrink<2>(box);
}

#endif
