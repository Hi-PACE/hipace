/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_GPUUTIL_H_
#define HIPACE_GPUUTIL_H_

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
