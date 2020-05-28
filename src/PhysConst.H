#ifndef HIPACE_CONST_H_
#define HIPACE_CONST_H_

#include <AMReX_REAL.H>

// Physical constants. Values are the 2018 CODATA recommended values
// https://physics.nist.gov/cuu/Constants/index.html
namespace PhysConst
{
    static constexpr amrex::Real c     = 299'792'458.;
    static constexpr amrex::Real ep0   = 8.8541878128e-12;
    static constexpr amrex::Real mu0   = 1.25663706212e-06;
    static constexpr amrex::Real q_e   = 1.602176634e-19;
    static constexpr amrex::Real m_e   = 9.1093837015e-31;
}

#endif
