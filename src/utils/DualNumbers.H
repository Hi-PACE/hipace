/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_DUALNUMBERS_H_
#define HIPACE_DUALNUMBERS_H_

#include <AMReX_Config.H>

struct DualNumber {
    amrex::Real value;
    amrex::Real epsilon;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr
    DualNumber (amrex::Real val, amrex::Real ep) : value{val}, epsilon{ep} {}
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr
    DualNumber (amrex::Real val) : value{val}, epsilon{0} {}
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr
    DualNumber () : value{0}, epsilon{0} {}

    friend AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr
    DualNumber operator+ (const DualNumber& a, const DualNumber& b) {
        return {a.value + b.value, a.epsilon + b.epsilon};
    }

    friend AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr
    DualNumber operator- (const DualNumber& a, const DualNumber& b) {
        return {a.value - b.value, a.epsilon - b.epsilon};
    }

    friend AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr
    DualNumber operator* (const DualNumber& a, const DualNumber& b) {
        return {a.value * b.value, a.epsilon * b.value + a.value * b.epsilon};
    }

    friend AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr
    DualNumber operator/ (const DualNumber& a, const DualNumber& b) {
        return {a.value / b.value, (a.epsilon * b.value - a.value * b.epsilon)/(b.value * b.value)};
    }
};

#endif
