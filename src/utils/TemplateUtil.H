/* Copyright 2023
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

    template <class T, std::size_t N>
    constexpr auto SingleTypeMultiplier(const T (&)[N]) {
        return SingleTypeMultiplier_imp<T, N>();
    }

    template <class T>
    constexpr auto SingleTypeMultiplier(T) {
        return amrex::TypeList<T>{};
    }

    template <template <class...> class TParam, class... Args>
    constexpr auto TApply(amrex::TypeList<Args...>) {
        return amrex::TypeList<TParam<Args...>>{};
    }

}

template <template <class...> class TParam, class... Types>
using TypeMultiplier = amrex::TypeAt<0, decltype(detail::TApply<TParam>(
    (amrex::TypeList<>{} + ... + detail::SingleTypeMultiplier(Types{}))
))>;


template<auto I, auto N, class F>
constexpr void constexpr_for (F&& f) // TODO: use amrex version
{
    if constexpr (I < N) {
        f(std::integral_constant<decltype(I), I>());
        constexpr_for<I+1, N>(f);
    }
}

#endif
