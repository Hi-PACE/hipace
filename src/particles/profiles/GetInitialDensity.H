#ifndef GETINTIALDENSITY_H_
#define GETINTIALDENSITY_H_

#include "particles/BeamParticleContainer.H"
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>

/** \brief Beam profile type, currently only Gaussian or Flattop */
enum struct BeamProfileType { Gaussian, Flattop };

struct GetInitialDensity
{
    GetInitialDensity () = default;

    GetInitialDensity (amrex::Real a_density); //, amrex::Real mean, amrex::Real std);

    amrex::Real operator() (amrex::Real x, amrex::Real y, amrex::Real z);
    amrex::RealVect  m_mean;
    amrex::RealVect  m_std {0.,0.,0.};
    BeamProfileType m_profile;
    amrex::Real m_density;
};

#endif // GETINTIALDENSITY_H_
