#ifndef GETINTIALDENSITY_H_
#define GETINTIALDENSITY_H_

#include <math.h>

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>

/** \brief Beam profile type, currently only Gaussian or Flattop */
enum struct BeamProfileType { Gaussian, Flattop };

struct GetInitialDensity
{
    GetInitialDensity () = default;

    GetInitialDensity (amrex::Real a_density);

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::Real x, amrex::Real y, amrex::Real z) const
    {
        using namespace amrex::literals;
        amrex::Real weight = 0._rt;
        if        (m_profile == BeamProfileType::Gaussian) {
            weight = m_density* exp(-0.5* pow( (x - m_mean[0]) / m_std[0], 2) )
                              * exp(-0.5* pow( (y - m_mean[1]) / m_std[1], 2) )
                              * exp(-0.5* pow( (z - m_mean[2]) / m_std[2], 2) );
        } else if (m_profile == BeamProfileType::Flattop) {
            weight = m_density;
        }
        return weight;
    }
    amrex::RealVect  m_mean {0.,0.,0.};
    amrex::RealVect  m_std;
    BeamProfileType m_profile;
    amrex::Real m_density;
};

#endif // GETINTIALDENSITY_H_
