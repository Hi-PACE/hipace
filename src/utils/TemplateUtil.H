/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_TEMPLATEUTIL_H_
#define HIPACE_TEMPLATEUTIL_H_

#include <AMReX_TypeList.H>
#include <AMReX_Tuple.H>

#include <type_traits>

// zero initialize a GpuTuple
template <class... Args>
constexpr auto MakeZeroTuple(amrex::GpuTuple<Args...>) {
    return amrex::GpuTuple<Args...>{Args{0}...};
}

namespace detail {

    // return TypeList<T, T, T, T, ... (N times)> by using the fast power algorithm
    template <class T, std::size_t N>
    constexpr auto SingleTypeMultiplier_imp() {
        if constexpr (N == 0) {
            return amrex::TypeList<>{};
        } else if constexpr (N == 1) {
            return amrex::TypeList<T>{};
        } else if constexpr (N % 2 == 0) {
            return SingleTypeMultiplier_imp<T, N / 2>() + SingleTypeMultiplier_imp<T, N / 2>();
        } else {
            return SingleTypeMultiplier_imp<T, N - 1>() + amrex::TypeList<T>{};
        }
    }

    // overload of SingleTypeMultiplier for multiple types:
    // convert T[N] to  T, T, T, T, ... (N times with N >= 1)
    template <class T, std::size_t N>
    constexpr auto SingleTypeMultiplier(const T (&)[N]) {
        return SingleTypeMultiplier_imp<T, N>();
    }

    // overload of SingleTypeMultiplier for one regular type
    template <class T>
    constexpr auto SingleTypeMultiplier(T) {
        return amrex::TypeList<T>{};
    }

    // apply the types of the input TypeList as template arguments to TParam
    template <template <class...> class TParam, class... Args>
    constexpr auto TApply(amrex::TypeList<Args...>) {
        return amrex::TypeList<TParam<Args...>>{};
    }

}

// Return the first template argument with the later arguments applied to it.
// Types of the form T[N] are expanded to T, T, T, T, ... (N times with N >= 1).
template <template <class...> class TParam, class... Types>
using TypeMultiplier = amrex::TypeAt<0, decltype(detail::TApply<TParam>(
    (amrex::TypeList<>{} + ... + detail::SingleTypeMultiplier(Types{}))
))>;

#endif
