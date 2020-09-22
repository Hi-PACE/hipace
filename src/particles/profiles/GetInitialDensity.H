#ifndef GETINTIALDENSITY_H_
#define GETINTIALDENSITY_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>

/** \brief Beam profile type, currently only Gaussian or Flattop */
enum struct BeamProfileType { Gaussian, Flattop };

struct GetInitialDensity
{
    GetInitialDensity () = default;

    GetInitialDensity (amrex::Real a_density); //, amrex::Real mean, amrex::Real std);

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::Real x, amrex::Real y, amrex::Real z) const
    {
        using namespace amrex::literals;
        amrex::Real weight = 0._rt;
        if        (m_profile == BeamProfileType::Gaussian){
            weight = x + y + z; // to be Gaussian
        } else if (m_profile == BeamProfileType::Flattop)
            weight = m_density; // scale factor for SI units missing!
        return weight;
    }
    amrex::RealVect  m_mean;
    amrex::RealVect  m_std {0.,0.,0.};
    BeamProfileType m_profile;
    amrex::Real m_density;
};

#endif // GETINTIALDENSITY_H_
