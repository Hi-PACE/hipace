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
#include <cmath>

struct shape_factor_result {
    amrex::Real factor;
    int cell;
};

inline static constexpr amrex::GpuTuple<
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 1>, 1>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 2>, 2>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 3>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>
> depos_prefactor {
    {{
        {       1.},
    }}, {{
        {      -1.,       1.},
        {       1.,       0.}
    }}, {{
        {    1./2.,      -1.,    1./2.},
        {      -1.,       1.,    1./2.},
        {    1./2.,       0.,       0.}
    }}, {{
        {   -1./6.,    1./2.,   -1./2.,    1./6.},
        {    1./2.,      -1.,       0.,    2./3.},
        {   -1./2.,    1./2.,    1./2.,    1./6.},
        {    1./6.,       0.,       0.,       0.}
    }}
};

#ifdef AMREX_USE_CUDA
AMREX_GPU_CONSTANT inline static constexpr auto depos_prefactor_g = depos_prefactor;
#endif

template<int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto get_depos_prefactor (int ix) {
#ifdef AMREX_USE_CUDA
    AMREX_IF_ON_DEVICE((return amrex::get<depos_order>(depos_prefactor_g)[ix];))
    AMREX_IF_ON_HOST((return amrex::get<depos_order>(depos_prefactor)[ix];))
#else
    return amrex::get<depos_order>(depos_prefactor)[ix];
#endif
}

/** \brief Compute a single shape factor and return the index of the cell where the particle writes.
 *
 * \tparam depos_order Order of the shape factor
 * \param[in] xmid exact position of the particle in index space
 * \param[in] ix index of the shape factor, must be 0 <= ix <= depos_order
 */
template<int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
shape_factor_result shape_factor (amrex::Real xmid, int ix) noexcept {

    if constexpr (depos_order % 2 == 0) {
        xmid += amrex::Real(0.5);
    }

    const amrex::Real xfloor = std::floor(xmid);
    constexpr int floor_offset = - depos_order / 2;
    const int xbegin = static_cast<int>(xfloor) + floor_offset;

    const amrex::Real xint = xmid-xfloor;
    const auto s = get_depos_prefactor<depos_order>(ix);

    amrex::Real res = s[0];
    for (int i=1; i<=depos_order; ++i) {
        res = res * xint + s[i];
    }
    return {res, xbegin + ix};
}

struct derivative_shape_factor_result {
    amrex::Real factor;
    amrex::Real dx_factor;
    int cell;
};

inline static constexpr amrex::GpuTuple<
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 1>, 1>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 1>, 2>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 2>, 3>,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 4>
> ds_0_prefactor {
    {{
        {       0.}
    }}, {{
        {       1.},
        {      -1.}
    }}, {{
        {      -1.,       1.},
        {       2.,      -1.},
        {      -1.,       0.}
    }}, {{
        {    1./2.,      -1.,    1./2.},
        {   -3./2.,       2.,       0.},
        {    3./2.,      -1.,   -1./2.},
        {   -1./2.,       0.,       0.}
    }}
};

inline static constexpr amrex::GpuTuple<
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 1>, 2>, 3>,
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 2>, 3>, 3>,
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 4>, 3>,
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 5>, 3>
> ds_1_prefactor {
    {{
        {{
            {       1.},
            {       0.}
        }}, {{
            {       0.},
            {       1.}
        }}, {{
            {       1.},
            {      -1.}
        }}
    }}, {{
        {{
            {      -1.,    1./2.},
            {       1.,    1./2.},
            {       0.,       0.}
        }}, {{
            {       0.,       0.},
            {      -1.,    3./2.},
            {       1.,   -1./2.}
        }}, {{
            {      -1.,       1.},
            {       2.,      -1.},
            {      -1.,       0.}
        }}
    }}, {{
        {{
            {    1./2.,   -1./2.,    1./8.},
            {      -1.,       0.,    3./4.},
            {    1./2.,    1./2.,    1./8.},
            {       0.,       0.,       0.}
        }}, {{
            {       0.,       0.,       0.},
            {    1./2.,   -3./2.,    9./8.},
            {      -1.,       2.,   -1./4.},
            {    1./2.,   -1./2.,    1./8.}
        }}, {{
            {    1./2.,      -1.,    1./2.},
            {   -3./2.,       2.,       0.},
            {    3./2.,      -1.,   -1./2.},
            {   -1./2.,       0.,       0.}
        }}
    }}, {{
        {{
            {   -1./6.,    1./4.,   -1./8.,   1./48.},
            {    1./2.,   -1./4.,   -5./8.,  23./48.},
            {   -1./2.,   -1./4.,    5./8.,  23./48.},
            {    1./6.,    1./4.,    1./8.,   1./48.},
            {       0.,       0.,       0.,       0.}
        }}, {{
            {       0.,       0.,       0.,       0.},
            {   -1./6.,    3./4.,   -9./8.,   9./16.},
            {    1./2.,   -7./4.,   11./8.,  17./48.},
            {   -1./2.,    5./4.,   -3./8.,   5./48.},
            {    1./6.,   -1./4.,    1./8.,  -1./48.}
        }}, {{
            {   -1./6.,    1./2.,   -1./2.,    1./6.},
            {    2./3.,   -3./2.,    1./2.,    1./2.},
            {      -1.,    3./2.,    1./2.,   -1./2.},
            {    2./3.,   -1./2.,   -1./2.,   -1./6.},
            {   -1./6.,       0.,       0.,       0.}
        }}
    }}
};

inline static constexpr amrex::GpuTuple<
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 1>, 3>, 2>,
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 2>, 4>, 2>,
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 5>, 2>,
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 6>, 2>
> ds_2_prefactor {
    {{
        {{
            {       0.},
            {       1.},
            {       0.}
        }}, {{
            {    1./2.},
            {       0.},
            {   -1./2.}
        }}
    }}, {{
        {{
            {       0.,       0.},
            {      -1.,       1.},
            {       1.,       0.},
            {       0.,       0.}
        }}, {{
            {   -1./2.,    1./2.},
            {    1./2.,       0.},
            {    1./2.,   -1./2.},
            {   -1./2.,       0.}
        }}
    }}, {{
        {{
            {       0.,       0.,       0.},
            {    1./2.,      -1.,    1./2.},
            {      -1.,       1.,    1./2.},
            {    1./2.,       0.,       0.},
            {       0.,       0.,       0.}
        }}, {{
            {    1./4.,   -1./2.,    1./4.},
            {   -1./2.,    1./2.,    1./4.},
            {       0.,    1./2.,   -1./4.},
            {    1./2.,   -1./2.,   -1./4.},
            {   -1./4.,       0.,       0.}
        }}
    }}, {{
        {{
            {       0.,       0.,       0.,       0.},
            {   -1./6.,    1./2.,   -1./2.,    1./6.},
            {    1./2.,      -1.,       0.,    2./3.},
            {   -1./2.,    1./2.,    1./2.,    1./6.},
            {    1./6.,       0.,       0.,       0.},
            {       0.,       0.,       0.,       0.}
        }}, {{
            {  -1./12.,    1./4.,   -1./4.,   1./12.},
            {    1./4.,   -1./2.,       0.,    1./3.},
            {   -1./6.,       0.,    1./2.,       0.},
            {   -1./6.,    1./2.,       0.,   -1./3.},
            {    1./4.,   -1./4.,   -1./4.,  -1./12.},
            {  -1./12.,       0.,       0.,       0.}
        }}
    }}
};

#ifdef AMREX_USE_CUDA
AMREX_GPU_CONSTANT inline static constexpr auto ds_0_prefactor_g = ds_0_prefactor;
AMREX_GPU_CONSTANT inline static constexpr auto ds_1_prefactor_g = ds_1_prefactor;
AMREX_GPU_CONSTANT inline static constexpr auto ds_2_prefactor_g = ds_2_prefactor;
#endif

template<int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto get_ds_0_prefactor (int ix) {
#ifdef AMREX_USE_CUDA
    AMREX_IF_ON_DEVICE((return amrex::get<depos_order>(ds_0_prefactor_g)[ix];))
    AMREX_IF_ON_HOST((return amrex::get<depos_order>(ds_0_prefactor)[ix];))
#else
    return amrex::get<depos_order>(ds_0_prefactor)[ix];
#endif
}

template<int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto get_ds_1_prefactor (int ix, int type) {
#ifdef AMREX_USE_CUDA
    AMREX_IF_ON_DEVICE((return amrex::get<depos_order>(ds_1_prefactor_g)[type][ix];))
    AMREX_IF_ON_HOST((return amrex::get<depos_order>(ds_1_prefactor)[type][ix];))
#else
    return amrex::get<depos_order>(ds_1_prefactor)[type][ix];
#endif
}

template<int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto get_ds_2_prefactor (int ix, int type) {
#ifdef AMREX_USE_CUDA
    AMREX_IF_ON_DEVICE((return amrex::get<depos_order>(ds_2_prefactor_g)[type][ix];))
    AMREX_IF_ON_HOST((return amrex::get<depos_order>(ds_2_prefactor)[type][ix];))
#else
    return amrex::get<depos_order>(ds_2_prefactor)[type][ix];
#endif
}


/** \brief Compute a single derivative shape factor and return the index of the cell
 * where the particle writes.
 *
 * \tparam derivative_type which type of derivative to use
 * \tparam depos_order Order of the shape factor
 * \param[in] xmid exact position of the particle in index space
 * \param[in] ix index of the shape factor, must be 0 <= ix <= (depos_order + derivative_type)
 */
template<int derivative_type, int depos_order> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
derivative_shape_factor_result derivative_shape_factor (amrex::Real xmid, int ix) noexcept {

    if constexpr ((depos_order + derivative_type) % 2 == 0) {
        xmid += amrex::Real(0.5);
    }

    const amrex::Real xfloor = std::floor(xmid);
    constexpr int floor_offset = - (depos_order + derivative_type) / 2;
    const int xbegin = static_cast<int>(xfloor) + floor_offset;

    const amrex::Real xint = xmid-xfloor;

    if constexpr (derivative_type==0) { // analytic derivative
        const auto s = get_depos_prefactor<depos_order>(ix);
        const auto ds = get_ds_0_prefactor<depos_order>(ix);

        amrex::Real res = s[0];
        amrex::Real resd = ds[0];
        for (int i=1; i<=depos_order; ++i) {
            res = res * xint + s[i];
        }
        for (int i=1; i<depos_order; ++i) {
            resd = resd * xint + ds[i];
        }
        return {res, resd, xbegin + ix};
    } else if constexpr (derivative_type==1) { // nodal derivative
        const auto s = xint < amrex::Real(0.5) ?
            get_ds_1_prefactor<depos_order>(ix, 0) :
            get_ds_1_prefactor<depos_order>(ix, 1);
        const auto ds = get_ds_1_prefactor<depos_order>(ix, 2);

        amrex::Real res = s[0];
        amrex::Real resd = ds[0];
        for (int i=1; i<=depos_order; ++i) {
            res = res * xint + s[i];
            resd = resd * xint + ds[i];
        }
        return {res, resd, xbegin + ix};
    } else { // centered derivative
        const auto s = get_ds_2_prefactor<depos_order>(ix, 0);
        const auto ds = get_ds_2_prefactor<depos_order>(ix, 1);

        amrex::Real res = s[0];
        amrex::Real resd = ds[0];
        for (int i=1; i<=depos_order; ++i) {
            res = res * xint + s[i];
            resd = resd * xint + ds[i];
        }
        return {res, resd, xbegin + ix};
    }
}

#endif // SHAPEFACTORS_H_
