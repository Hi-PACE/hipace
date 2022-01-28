#ifndef OPEN_BOUNDARY_H_
#define OPEN_BOUNDARY_H_

#include <AMReX_AmrCore.H>
#include <cmath>

template<unsigned int exp> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pow (amrex::Real base) {
    return pow<exp-1>(base) * base;
}

template<> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pow<0> (amrex::Real base) {
    return 1;
}

template<> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pow<1> (amrex::Real base) {
    return base;
}

template<class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T GetMultipoleCoeffs (amrex::Real s_v, amrex::Real x, amrex::Real y)
{
    using namespace amrex::literals;
    return {
    s_v,
    s_v*x,
    s_v*y,
    s_v*(-pow<2>(x) + pow<2>(y)),
    s_v*x*y,
    s_v*(pow<3>(x) - 3*x*pow<2>(y)),
    s_v*(-3*pow<2>(x)*y + pow<3>(y)),
    s_v*(pow<4>(x) - 6*pow<2>(x)*pow<2>(y) + pow<4>(y)),
    s_v*(-pow<3>(x)*y + x*pow<3>(y)),
    s_v*(pow<5>(x) - 10*pow<3>(x)*pow<2>(y) + 5*x*pow<4>(y)),
    s_v*(5*pow<4>(x)*y - 10*pow<2>(x)*pow<3>(y) + pow<5>(y)),
    s_v*(-pow<6>(x) + 15*pow<4>(x)*pow<2>(y) - 15*pow<2>(x)*pow<4>(y) + pow<6>(y)),
    s_v*(pow<5>(x)*y - 10.0_rt/3.0_rt*pow<3>(x)*pow<3>(y) + x*pow<5>(y)),
    s_v*(pow<7>(x) - 21*pow<5>(x)*pow<2>(y) + 35*pow<3>(x)*pow<4>(y) - 7*x*pow<6>(y)),
    s_v*(-7*pow<6>(x)*y + 35*pow<4>(x)*pow<3>(y) - 21*pow<2>(x)*pow<5>(y) + pow<7>(y)),
    s_v*(pow<8>(x) - 28*pow<6>(x)*pow<2>(y) + 70*pow<4>(x)*pow<4>(y) - 28*pow<2>(x)*pow<6>(y) + pow<8>(y)),
    s_v*(-pow<7>(x)*y + 7*pow<5>(x)*pow<3>(y) - 7*pow<3>(x)*pow<5>(y) + x*pow<7>(y)),
    s_v*(pow<9>(x) - 36*pow<7>(x)*pow<2>(y) + 126*pow<5>(x)*pow<4>(y) - 84*pow<3>(x)*pow<6>(y) + 9*x*pow<8>(y)),
    s_v*(9*pow<8>(x)*y - 84*pow<6>(x)*pow<3>(y) + 126*pow<4>(x)*pow<5>(y) - 36*pow<2>(x)*pow<7>(y) + pow<9>(y)),
    s_v*(-pow<10>(x) + 45*pow<8>(x)*pow<2>(y) - 210*pow<6>(x)*pow<4>(y) + 210*pow<4>(x)*pow<6>(y) - 45*pow<2>(x)*pow<8>(y) + pow<10>(y)),
    s_v*(pow<9>(x)*y - 12*pow<7>(x)*pow<3>(y) + (126.0_rt/5.0_rt)*pow<5>(x)*pow<5>(y) - 12*pow<3>(x)*pow<7>(y) + x*pow<9>(y)),
    s_v*(pow<11>(x) - 55*pow<9>(x)*pow<2>(y) + 330*pow<7>(x)*pow<4>(y) - 462*pow<5>(x)*pow<6>(y) + 165*pow<3>(x)*pow<8>(y) - 11*x*pow<10>(y)),
    s_v*(-11*pow<10>(x)*y + 165*pow<8>(x)*pow<3>(y) - 462*pow<6>(x)*pow<5>(y) + 330*pow<4>(x)*pow<7>(y) - 55*pow<2>(x)*pow<9>(y) + pow<11>(y)),
    s_v*(pow<12>(x) - 66*pow<10>(x)*pow<2>(y) + 495*pow<8>(x)*pow<4>(y) - 924*pow<6>(x)*pow<6>(y) + 495*pow<4>(x)*pow<8>(y) - 66*pow<2>(x)*pow<10>(y) + pow<12>(y)),
    s_v*(-pow<11>(x)*y + (55.0_rt/3.0_rt)*pow<9>(x)*pow<3>(y) - 66*pow<7>(x)*pow<5>(y) + 66*pow<5>(x)*pow<7>(y) - 55.0_rt/3.0_rt*pow<3>(x)*pow<9>(y) + x*pow<11>(y)),
    s_v*(pow<13>(x) - 78*pow<11>(x)*pow<2>(y) + 715*pow<9>(x)*pow<4>(y) - 1716*pow<7>(x)*pow<6>(y) + 1287*pow<5>(x)*pow<8>(y) - 286*pow<3>(x)*pow<10>(y) + 13*x*pow<12>(y)),
    s_v*(13*pow<12>(x)*y - 286*pow<10>(x)*pow<3>(y) + 1287*pow<8>(x)*pow<5>(y) - 1716*pow<6>(x)*pow<7>(y) + 715*pow<4>(x)*pow<9>(y) - 78*pow<2>(x)*pow<11>(y) + pow<13>(y)),
    s_v*(-pow<14>(x) + 91*pow<12>(x)*pow<2>(y) - 1001*pow<10>(x)*pow<4>(y) + 3003*pow<8>(x)*pow<6>(y) - 3003*pow<6>(x)*pow<8>(y) + 1001*pow<4>(x)*pow<10>(y) - 91*pow<2>(x)*pow<12>(y) + pow<14>(y)),
    s_v*(pow<13>(x)*y - 26*pow<11>(x)*pow<3>(y) + 143*pow<9>(x)*pow<5>(y) - 1716.0_rt/7.0_rt*pow<7>(x)*pow<7>(y) + 143*pow<5>(x)*pow<9>(y) - 26*pow<3>(x)*pow<11>(y) + x*pow<13>(y)),
    s_v*(pow<15>(x) - 105*pow<13>(x)*pow<2>(y) + 1365*pow<11>(x)*pow<4>(y) - 5005*pow<9>(x)*pow<6>(y) + 6435*pow<7>(x)*pow<8>(y) - 3003*pow<5>(x)*pow<10>(y) + 455*pow<3>(x)*pow<12>(y) - 15*x*pow<14>(y)),
    s_v*(-15*pow<14>(x)*y + 455*pow<12>(x)*pow<3>(y) - 3003*pow<10>(x)*pow<5>(y) + 6435*pow<8>(x)*pow<7>(y) - 5005*pow<6>(x)*pow<9>(y) + 1365*pow<4>(x)*pow<11>(y) - 105*pow<2>(x)*pow<13>(y) + pow<15>(y)),
    s_v*(pow<16>(x) - 120*pow<14>(x)*pow<2>(y) + 1820*pow<12>(x)*pow<4>(y) - 8008*pow<10>(x)*pow<6>(y) + 12870*pow<8>(x)*pow<8>(y) - 8008*pow<6>(x)*pow<10>(y) + 1820*pow<4>(x)*pow<12>(y) - 120*pow<2>(x)*pow<14>(y) + pow<16>(y)),
    s_v*(-pow<15>(x)*y + 35*pow<13>(x)*pow<3>(y) - 273*pow<11>(x)*pow<5>(y) + 715*pow<9>(x)*pow<7>(y) - 715*pow<7>(x)*pow<9>(y) + 273*pow<5>(x)*pow<11>(y) - 35*pow<3>(x)*pow<13>(y) + x*pow<15>(y)),
    s_v*(pow<17>(x) - 136*pow<15>(x)*pow<2>(y) + 2380*pow<13>(x)*pow<4>(y) - 12376*pow<11>(x)*pow<6>(y) + 24310*pow<9>(x)*pow<8>(y) - 19448*pow<7>(x)*pow<10>(y) + 6188*pow<5>(x)*pow<12>(y) - 680*pow<3>(x)*pow<14>(y) + 17*x*pow<16>(y)),
    s_v*(17*pow<16>(x)*y - 680*pow<14>(x)*pow<3>(y) + 6188*pow<12>(x)*pow<5>(y) - 19448*pow<10>(x)*pow<7>(y) + 24310*pow<8>(x)*pow<9>(y) - 12376*pow<6>(x)*pow<11>(y) + 2380*pow<4>(x)*pow<13>(y) - 136*pow<2>(x)*pow<15>(y) + pow<17>(y)),
    s_v*(-pow<18>(x) + 153*pow<16>(x)*pow<2>(y) - 3060*pow<14>(x)*pow<4>(y) + 18564*pow<12>(x)*pow<6>(y) - 43758*pow<10>(x)*pow<8>(y) + 43758*pow<8>(x)*pow<10>(y) - 18564*pow<6>(x)*pow<12>(y) + 3060*pow<4>(x)*pow<14>(y) - 153*pow<2>(x)*pow<16>(y) + pow<18>(y)),
    s_v*(pow<17>(x)*y - 136.0_rt/3.0_rt*pow<15>(x)*pow<3>(y) + 476*pow<13>(x)*pow<5>(y) - 1768*pow<11>(x)*pow<7>(y) + (24310.0_rt/9.0_rt)*pow<9>(x)*pow<9>(y) - 1768*pow<7>(x)*pow<11>(y) + 476*pow<5>(x)*pow<13>(y) - 136.0_rt/3.0_rt*pow<3>(x)*pow<15>(y) + x*pow<17>(y))
    };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real GetFieldMultipole (amrex::Real* m_c, amrex::Real x_domain, amrex::Real y_domain)
{
    using namespace amrex::literals;
    amrex::Real radius_2 = pow<2>(x_domain) + pow<2>(y_domain);
    amrex::Real x = x_domain / radius_2;
    amrex::Real y = y_domain / radius_2;
    return
      m_c[0] * (std::log(radius_2))
    + m_c[1] * (-2*x)
    + m_c[2] * (-2*y)
    + m_c[3] * (pow<2>(x) - pow<2>(y))
    + m_c[4] * (-4*x*y)
    + m_c[5] * (-2.0_rt/3.0_rt*pow<3>(x) + 2*x*pow<2>(y))
    + m_c[6] * (2*pow<2>(x)*y - 2.0_rt/3.0_rt*pow<3>(y))
    + m_c[7] * (-1.0_rt/2.0_rt*pow<4>(x) + 3*pow<2>(x)*pow<2>(y) - 1.0_rt/2.0_rt*pow<4>(y))
    + m_c[8] * (8*pow<3>(x)*y - 8*x*pow<3>(y))
    + m_c[9] * (-2.0_rt/5.0_rt*pow<5>(x) + 4*pow<3>(x)*pow<2>(y) - 2*x*pow<4>(y))
    + m_c[10] * (-2*pow<4>(x)*y + 4*pow<2>(x)*pow<3>(y) - 2.0_rt/5.0_rt*pow<5>(y))
    + m_c[11] * ((1.0_rt/3.0_rt)*pow<6>(x) - 5*pow<4>(x)*pow<2>(y) + 5*pow<2>(x)*pow<4>(y) - 1.0_rt/3.0_rt*pow<6>(y))
    + m_c[12] * (-12*pow<5>(x)*y + 40*pow<3>(x)*pow<3>(y) - 12*x*pow<5>(y))
    + m_c[13] * (-2.0_rt/7.0_rt*pow<7>(x) + 6*pow<5>(x)*pow<2>(y) - 10*pow<3>(x)*pow<4>(y) + 2*x*pow<6>(y))
    + m_c[14] * (2*pow<6>(x)*y - 10*pow<4>(x)*pow<3>(y) + 6*pow<2>(x)*pow<5>(y) - 2.0_rt/7.0_rt*pow<7>(y))
    + m_c[15] * (-1.0_rt/4.0_rt*pow<8>(x) + 7*pow<6>(x)*pow<2>(y) - 35.0_rt/2.0_rt*pow<4>(x)*pow<4>(y) + 7*pow<2>(x)*pow<6>(y) - 1.0_rt/4.0_rt*pow<8>(y))
    + m_c[16] * (16*pow<7>(x)*y - 112*pow<5>(x)*pow<3>(y) + 112*pow<3>(x)*pow<5>(y) - 16*x*pow<7>(y))
    + m_c[17] * (-2.0_rt/9.0_rt*pow<9>(x) + 8*pow<7>(x)*pow<2>(y) - 28*pow<5>(x)*pow<4>(y) + (56.0_rt/3.0_rt)*pow<3>(x)*pow<6>(y) - 2*x*pow<8>(y))
    + m_c[18] * (-2*pow<8>(x)*y + (56.0_rt/3.0_rt)*pow<6>(x)*pow<3>(y) - 28*pow<4>(x)*pow<5>(y) + 8*pow<2>(x)*pow<7>(y) - 2.0_rt/9.0_rt*pow<9>(y))
    + m_c[19] * ((1.0_rt/5.0_rt)*pow<10>(x) - 9*pow<8>(x)*pow<2>(y) + 42*pow<6>(x)*pow<4>(y) - 42*pow<4>(x)*pow<6>(y) + 9*pow<2>(x)*pow<8>(y) - 1.0_rt/5.0_rt*pow<10>(y))
    + m_c[20] * (-20*pow<9>(x)*y + 240*pow<7>(x)*pow<3>(y) - 504*pow<5>(x)*pow<5>(y) + 240*pow<3>(x)*pow<7>(y) - 20*x*pow<9>(y))
    + m_c[21] * (-2.0_rt/11.0_rt*pow<11>(x) + 10*pow<9>(x)*pow<2>(y) - 60*pow<7>(x)*pow<4>(y) + 84*pow<5>(x)*pow<6>(y) - 30*pow<3>(x)*pow<8>(y) + 2*x*pow<10>(y))
    + m_c[22] * (2*pow<10>(x)*y - 30*pow<8>(x)*pow<3>(y) + 84*pow<6>(x)*pow<5>(y) - 60*pow<4>(x)*pow<7>(y) + 10*pow<2>(x)*pow<9>(y) - 2.0_rt/11.0_rt*pow<11>(y))
    + m_c[23] * (-1.0_rt/6.0_rt*pow<12>(x) + 11*pow<10>(x)*pow<2>(y) - 165.0_rt/2.0_rt*pow<8>(x)*pow<4>(y) + 154*pow<6>(x)*pow<6>(y) - 165.0_rt/2.0_rt*pow<4>(x)*pow<8>(y) + 11*pow<2>(x)*pow<10>(y) - 1.0_rt/6.0_rt*pow<12>(y))
    + m_c[24] * (24*pow<11>(x)*y - 440*pow<9>(x)*pow<3>(y) + 1584*pow<7>(x)*pow<5>(y) - 1584*pow<5>(x)*pow<7>(y) + 440*pow<3>(x)*pow<9>(y) - 24*x*pow<11>(y))
    + m_c[25] * (-2.0_rt/13.0_rt*pow<13>(x) + 12*pow<11>(x)*pow<2>(y) - 110*pow<9>(x)*pow<4>(y) + 264*pow<7>(x)*pow<6>(y) - 198*pow<5>(x)*pow<8>(y) + 44*pow<3>(x)*pow<10>(y) - 2*x*pow<12>(y))
    + m_c[26] * (-2*pow<12>(x)*y + 44*pow<10>(x)*pow<3>(y) - 198*pow<8>(x)*pow<5>(y) + 264*pow<6>(x)*pow<7>(y) - 110*pow<4>(x)*pow<9>(y) + 12*pow<2>(x)*pow<11>(y) - 2.0_rt/13.0_rt*pow<13>(y))
    + m_c[27] * ((1.0_rt/7.0_rt)*pow<14>(x) - 13*pow<12>(x)*pow<2>(y) + 143*pow<10>(x)*pow<4>(y) - 429*pow<8>(x)*pow<6>(y) + 429*pow<6>(x)*pow<8>(y) - 143*pow<4>(x)*pow<10>(y) + 13*pow<2>(x)*pow<12>(y) - 1.0_rt/7.0_rt*pow<14>(y))
    + m_c[28] * (-28*pow<13>(x)*y + 728*pow<11>(x)*pow<3>(y) - 4004*pow<9>(x)*pow<5>(y) + 6864*pow<7>(x)*pow<7>(y) - 4004*pow<5>(x)*pow<9>(y) + 728*pow<3>(x)*pow<11>(y) - 28*x*pow<13>(y))
    + m_c[29] * (-2.0_rt/15.0_rt*pow<15>(x) + 14*pow<13>(x)*pow<2>(y) - 182*pow<11>(x)*pow<4>(y) + (2002.0_rt/3.0_rt)*pow<9>(x)*pow<6>(y) - 858*pow<7>(x)*pow<8>(y) + (2002.0_rt/5.0_rt)*pow<5>(x)*pow<10>(y) - 182.0_rt/3.0_rt*pow<3>(x)*pow<12>(y) + 2*x*pow<14>(y))
    + m_c[30] * (2*pow<14>(x)*y - 182.0_rt/3.0_rt*pow<12>(x)*pow<3>(y) + (2002.0_rt/5.0_rt)*pow<10>(x)*pow<5>(y) - 858*pow<8>(x)*pow<7>(y) + (2002.0_rt/3.0_rt)*pow<6>(x)*pow<9>(y) - 182*pow<4>(x)*pow<11>(y) + 14*pow<2>(x)*pow<13>(y) - 2.0_rt/15.0_rt*pow<15>(y))
    + m_c[31] * (-1.0_rt/8.0_rt*pow<16>(x) + 15*pow<14>(x)*pow<2>(y) - 455.0_rt/2.0_rt*pow<12>(x)*pow<4>(y) + 1001*pow<10>(x)*pow<6>(y) - 6435.0_rt/4.0_rt*pow<8>(x)*pow<8>(y) + 1001*pow<6>(x)*pow<10>(y) - 455.0_rt/2.0_rt*pow<4>(x)*pow<12>(y) + 15*pow<2>(x)*pow<14>(y) - 1.0_rt/8.0_rt*pow<16>(y))
    + m_c[32] * (32*pow<15>(x)*y - 1120*pow<13>(x)*pow<3>(y) + 8736*pow<11>(x)*pow<5>(y) - 22880*pow<9>(x)*pow<7>(y) + 22880*pow<7>(x)*pow<9>(y) - 8736*pow<5>(x)*pow<11>(y) + 1120*pow<3>(x)*pow<13>(y) - 32*x*pow<15>(y))
    + m_c[33] * (-2.0_rt/17.0_rt*pow<17>(x) + 16*pow<15>(x)*pow<2>(y) - 280*pow<13>(x)*pow<4>(y) + 1456*pow<11>(x)*pow<6>(y) - 2860*pow<9>(x)*pow<8>(y) + 2288*pow<7>(x)*pow<10>(y) - 728*pow<5>(x)*pow<12>(y) + 80*pow<3>(x)*pow<14>(y) - 2*x*pow<16>(y))
    + m_c[34] * (-2*pow<16>(x)*y + 80*pow<14>(x)*pow<3>(y) - 728*pow<12>(x)*pow<5>(y) + 2288*pow<10>(x)*pow<7>(y) - 2860*pow<8>(x)*pow<9>(y) + 1456*pow<6>(x)*pow<11>(y) - 280*pow<4>(x)*pow<13>(y) + 16*pow<2>(x)*pow<15>(y) - 2.0_rt/17.0_rt*pow<17>(y))
    + m_c[35] * ((1.0_rt/9.0_rt)*pow<18>(x) - 17*pow<16>(x)*pow<2>(y) + 340*pow<14>(x)*pow<4>(y) - 6188.0_rt/3.0_rt*pow<12>(x)*pow<6>(y) + 4862*pow<10>(x)*pow<8>(y) - 4862*pow<8>(x)*pow<10>(y) + (6188.0_rt/3.0_rt)*pow<6>(x)*pow<12>(y) - 340*pow<4>(x)*pow<14>(y) + 17*pow<2>(x)*pow<16>(y) - 1.0_rt/9.0_rt*pow<18>(y))
    + m_c[36] * (-36*pow<17>(x)*y + 1632*pow<15>(x)*pow<3>(y) - 17136*pow<13>(x)*pow<5>(y) + 63648*pow<11>(x)*pow<7>(y) - 97240*pow<9>(x)*pow<9>(y) + 63648*pow<7>(x)*pow<11>(y) - 17136*pow<5>(x)*pow<13>(y) + 1632*pow<3>(x)*pow<15>(y) - 36*x*pow<17>(y))
    ;
}

#endif
