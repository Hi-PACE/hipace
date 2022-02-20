#ifndef HIPACE_COMPUTE_TEMPERATURE_H_
#define HIPACE_COMPUTE_TEMPERATURE_H_

template <typename T_index, typename T_R>
T_R ComputeTemperature (
    T_index const Is, T_index const Ie, T_index const *I,
    T_R const *ux, T_R const *uy, T_R const *psi, T_R const m )
{
    using namespace amrex::literals;
    T_R constexpr c2i = 1._rt/(PhysConstSI::c * PhysConstSI::c);

    int N = Ie - Is;
    if ( N == 0 ) { return T_R(0.0); }

    T_R vx = T_R(0.0);    T_R vy = T_R(0.0);
    T_R vz = T_R(0.0);    T_R vs = T_R(0.0);
    T_R gm = T_R(0.0);    T_R us = T_R(0.0);

    for (int i = Is; i < Ie; ++i)
    {
        // particle's normalized vector potential
        const amrex::Real psii = psi[I[i]] *
            PhysConstSI::q_e / (PhysConstSI::m_e * PhysConstSI::c * PhysConstSI::c);
        // particle's Lorentz factor
        gm = (1.0_rt + ux[I[i]]*ux[I[i]]*c2i + uy[I[i]]*uy[I[i]]*c2i
              + (psii+1._rt)*(psii+1._rt)) / (2.0_rt * (psii+1._rt) );
        const amrex::Real uz = PhysConstSI::c * (gm - psii - 1._rt);
        us = ( ux[ I[i] ] * ux[ I[i] ] +
               uy[ I[i] ] * uy[ I[i] ] +
               uz         * uz);
        // gm = std::sqrt( T_R(1.0) + us*c2i );
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
