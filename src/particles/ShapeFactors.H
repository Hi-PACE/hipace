/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Remi Lehe
 * Severin Diederichs, WeiqunZhang
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2019 Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */


#ifndef SHAPEFACTORS_H_
#define SHAPEFACTORS_H_

#include <AMReX_Math.H>

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialized templates are defined below for orders 0 to 3.
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int compute_shape_factor (amrex::Real* const sx, amrex::Real xint)
{
    amrex::ignore_unused(sx, xint);
    return 0;
}

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 0
 */
template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int compute_shape_factor <0> (amrex::Real* const sx, amrex::Real xmid){

    using namespace amrex::literals;

    const auto j = static_cast<int>(amrex::Math::floor(xmid+0.5_rt));
    sx[0] = 1.0_rt;
    return j;
}

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 1
 */
template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int compute_shape_factor <1> (amrex::Real* const sx, amrex::Real xmid){

    using namespace amrex::literals;

    const auto j = static_cast<int>(amrex::Math::floor(xmid));
    const amrex::Real xint = xmid-j;
    sx[0] = 1.0_rt - xint;
    sx[1] = xint;
    return j;
}

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 2
 */
template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int compute_shape_factor <2> (amrex::Real* const sx, amrex::Real xmid){

    using namespace amrex::literals;

    const auto j = static_cast<int>(amrex::Math::floor(xmid+0.5_rt));
    const amrex::Real xint = xmid-j;
    sx[0] = 0.5_rt*(0.5_rt-xint)*(0.5_rt-xint);
    sx[1] = 0.75_rt-xint*xint;
    sx[2] = 0.5_rt*(0.5_rt+xint)*(0.5_rt+xint);
    // index of the leftmost cell where particle deposits
    return j-1;
}

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 3
 */
template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int compute_shape_factor <3> (amrex::Real* const sx, amrex::Real xmid){

    using namespace amrex::literals;

    const auto j = static_cast<int>(amrex::Math::floor(xmid));
    const amrex::Real xint = xmid-j;
    sx[0] = 1.0_rt/6.0_rt*(1.0_rt-xint)*(1.0_rt-xint)*(1.0_rt-xint);
    sx[1] = 2.0_rt/3.0_rt-xint*xint*(1.0_rt-xint/2.0_rt);
    sx[2] = 2.0_rt/3.0_rt-(1.0_rt-xint)*(1.0_rt-xint)*(1.0_rt-0.5_rt*(1.0_rt-xint));
    sx[3] = 1.0_rt/6.0_rt*xint*xint*xint;
    // index of the leftmost cell where particle deposits
    return j-1;
}

struct shape_factor_result {
    amrex::Real factor;
    int cell;
};

/** \brief Compute a single shape factor and return the index of the cell where the particle writes.
 *
 * \tparam branchless whether to use a branchless calculation or one with branches depending in ix
 * \tparam depos_order Order of the shape factor
 * \param[in] xmid exact position of the particle in index space
 * \param[in] ix index of the shape factor, must be 0 <= ix <= depos_order
 */
template<bool branchless, int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
shape_factor_result compute_single_shape_factor (amrex::Real xmid, int ix) noexcept {

    using namespace amrex::literals;
    if constexpr (branchless) {
        if constexpr (depos_order==0) {
            return {1._rt, static_cast<int>(amrex::Math::floor(xmid+0.5_rt))};
        } else if constexpr (depos_order==1) {
            const amrex::Real xfloor = amrex::Math::floor(xmid);
            const amrex::Real xint = xmid-xfloor;
            const amrex::Real ic = static_cast<amrex::Real>(ix);
            const amrex::Real shape = (1._rt-ic) + (2._rt*ic-1._rt) * xint;
            return {shape, static_cast<int>(xfloor)+ix};
        } else if constexpr (depos_order==2) {
            const amrex::Real xfloor = amrex::Math::floor(xmid+0.5_rt);
            const amrex::Real xint = xmid-xfloor;
            const int ic = ix - 1;
            const amrex::Real icr = static_cast<amrex::Real>(ic);
            const amrex::Real shape = (-0.625_rt*icr*icr + 0.75_rt)
                                    + (0.5_rt*icr + 1.5_rt*icr*icr*xint - xint) * xint;
            return {shape, static_cast<int>(xfloor)+ic};
        } else if constexpr (depos_order==3) {
            const amrex::Real xfloor = amrex::Math::floor(xmid);
            const amrex::Real xint = xmid-xfloor;
            const int ic = ix + 1;
            const amrex::Real icr = static_cast<amrex::Real>(ic&1);
            const amrex::Real xint_s = icr + (1._rt - 2._rt*icr) * xint;
            const amrex::Real cond = static_cast<amrex::Real>(ic&2);
            const amrex::Real shape = (1.0_rt/6.0_rt)*(cond*2._rt
                                    + xint_s*xint_s*(cond*(xint_s-3._rt) + xint_s));
            return {shape, static_cast<int>(xfloor)+ix-1};
        }
    } else {
        if constexpr (depos_order==0) {
            return {1._rt, static_cast<int>(amrex::Math::floor(xmid+0.5_rt))};
        } else if constexpr (depos_order==1) {
            const amrex::Real xfloor = amrex::Math::floor(xmid);
            const amrex::Real xint = xmid-xfloor;
            if (ix==0) {
                return {1.0_rt - xint, static_cast<int>(xfloor)};
            } else {
                return {xint, static_cast<int>(xfloor)+1};
            }
        } else if constexpr (depos_order==2) {
            const amrex::Real xfloor = amrex::Math::floor(xmid+0.5_rt);
            const amrex::Real xint = xmid-xfloor;
            if (ix==0) {
                return {0.5_rt*(0.5_rt-xint)*(0.5_rt-xint), static_cast<int>(xfloor)-1};
            } else if (ix==1) {
                return {0.75_rt-xint*xint, static_cast<int>(xfloor)};
            } else {
                return {0.5_rt*(0.5_rt+xint)*(0.5_rt+xint), static_cast<int>(xfloor)+1};
            }
        } else if constexpr (depos_order==3) {
            const amrex::Real xfloor = amrex::Math::floor(xmid);
            const amrex::Real xint = xmid-xfloor;
            if (ix==0) {
                return {1.0_rt/6.0_rt*(1.0_rt-xint)*(1.0_rt-xint)*(1.0_rt-xint),
                            static_cast<int>(xfloor)-1};
            } else if (ix==1) {
                return {2.0_rt/3.0_rt-xint*xint*(1.0_rt-xint/2.0_rt),
                            static_cast<int>(xfloor)};
            } else if (ix==2) {
                return {2.0_rt/3.0_rt-(1.0_rt-xint)*(1.0_rt-xint)*(1.0_rt-0.5_rt*(1.0_rt-xint)),
                            static_cast<int>(xfloor)+1};
            } else {
                return {1.0_rt/6.0_rt*xint*xint*xint,
                            static_cast<int>(xfloor)+2};
            }
        }
    }
    static_assert(0 <= depos_order && depos_order <= 3);
    return {};
}

#endif // SHAPEFACTORS_H_
