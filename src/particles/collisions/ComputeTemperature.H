/* Copyright 2019-2020 Andrew Myers, Yinjian Zhao, Maxence Thevenet, Severin Diederichs
 *
 * This file is part of HiPACE++ and WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef HIPACE_COMPUTE_TEMPERATURE_H_
#define HIPACE_COMPUTE_TEMPERATURE_H_

template <typename T_index, typename T_R>
AMREX_GPU_HOST_DEVICE
T_R ComputeTemperature (
    T_index const Is, T_index const Ie, T_index const * AMREX_RESTRICT I,
    T_R const * AMREX_RESTRICT ux, T_R const * AMREX_RESTRICT uy, T_R const * AMREX_RESTRICT psi,
    T_R const m, T_R const clight, T_R const inv_c2, bool is_beam_coll )
{
    using namespace amrex::literals;

    const int N = Ie - Is;
    if ( N == 0 ) { return T_R(0.0); }

    T_R vx = T_R(0.0);    T_R vy = T_R(0.0);
    T_R vz = T_R(0.0);    T_R vs = T_R(0.0);
    T_R gm = T_R(0.0);    T_R us = T_R(0.0);

    for (int i = Is; i < (int) Ie; ++i)
    {
        // particle's Lorentz factor
        gm = is_beam_coll ? std::sqrt( 1._rt + (ux[I[i]]*ux[I[i]] + uy[I[i]]*uy[I[i]]
                                                + psi[I[i]]*psi[I[i]])*inv_c2 )
                          : (1.0_rt + (ux[I[i]]*ux[I[i]] + uy[I[i]]*uy[I[i]])*inv_c2
                                       + psi[I[i]]*psi[I[i]]) / (2.0_rt * psi[I[i]] );
        const amrex::Real uz = is_beam_coll ? psi[I[i]] : clight * (gm - psi[I[i]]);
        us = ( ux[ I[i] ] * ux[ I[i] ] +
               uy[ I[i] ] * uy[ I[i] ] +
               uz         * uz);
        vx += ux[ I[i] ] / gm;
        vy += uy[ I[i] ] / gm;
        vz += uz         / gm;
        vs += us / gm / gm;
    }

    vx = vx / N;    vy = vy / N;
    vz = vz / N;    vs = vs / N;

    return m/T_R(3.0)*(vs-(vx*vx+vy*vy+vz*vz));
}

#endif // HIPACE_COMPUTE_TEMPERATURE_H_
