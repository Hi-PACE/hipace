#ifndef HIPACE_Constants_H_
#define HIPACE_Constants_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

namespace PhysConstSI
{
    static constexpr auto c   = static_cast<amrex::Real>( 299'792'458. );
    static constexpr auto ep0 = static_cast<amrex::Real>( 8.8541878128e-12 );
    static constexpr auto mu0 = static_cast<amrex::Real>( 1.25663706212e-06 );
    static constexpr auto q_e = static_cast<amrex::Real>( 1.602176634e-19 );
    static constexpr auto m_e = static_cast<amrex::Real>( 9.1093837015e-31 );
    static constexpr auto m_p = static_cast<amrex::Real>( 1.67262192369e-27 );
    static constexpr auto hbar  = static_cast<amrex::Real>( 1.054571817e-34 );
}

/** \brief Namespace containing math constants */
namespace MathConst
{
    static constexpr amrex::Real pi = 3.14159265358979323846;
}

/** \brief Struct containing physical constants, our main strategy to handle both SI and
 * normalized units.
 *
 * Cannot be a namespace with "static constexpr"s because these values depend on the unit system
 * (SI or normalized), which is defined at runtime.
 */
struct PhysConst
{
    amrex::Real c;
    amrex::Real ep0;
    amrex::Real mu0;
    amrex::Real q_e;
    amrex::Real m_e;
    amrex::Real m_p;
};

/** \brief SI Physical constants.
 *
 * Values are the 2018 CODATA recommended values
 * https://physics.nist.gov/cuu/Constants/index.html
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PhysConst make_constants_SI ()
{
    PhysConst const constants = {
        /* c   */ 299'792'458.,
        /* ep0 */ 8.8541878128e-12,
        /* mu0 */ 1.25663706212e-06,
        /* q_e */ 1.602176634e-19,
        /* m_e */ 9.1093837015e-31,
        /* m_p */ 1.67262192369e-27
    };
    return constants;
}

/** \brief Physical constants in normalized units */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PhysConst make_constants_normalized ()
{
    PhysConst const constants = {
        /* c   */ 1.,
        /* ep0 */ 1.,
        /* mu0 */ 1.,
        /* q_e */ 1.,
        /* m_e */ 1.,
        /* m_p */ 1836.15267343
    };
    return constants;
}
/** \brief Return the PhysConst object, which is a member of class Hipace */
PhysConst get_phys_const ();
#endif
