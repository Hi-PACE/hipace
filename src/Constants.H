#ifndef HIPACE_Constants_H_
#define HIPACE_Constants_H_

#include "Hipace.H"
#include <AMReX_REAL.H>
#include <cmath>

// Math constants
namespace MathConst
{
    static constexpr amrex::Real pi = 3.14159265358979323846;
}

// Physical constants. Values are the 2018 CODATA recommended values
// https://physics.nist.gov/cuu/Constants/index.html
namespace PhysConst
{
    static const amrex::Real c     = Hipace::m_normalized_units ? 1 : 299'792'458.;
    static const amrex::Real ep0   = Hipace::m_normalized_units ? 1 : 8.8541878128e-12;
    static const amrex::Real mu0   = Hipace::m_normalized_units ? 1 : 1.25663706212e-06;
    static const amrex::Real q_e   = Hipace::m_normalized_units ? 1 : 1.602176634e-19;
    static const amrex::Real m_e   = Hipace::m_normalized_units ? 1 : 9.1093837015e-31;
    static const amrex::Real m_p   = Hipace::m_normalized_units ? 1 : 1.67262192369e-27;
}

#endif
