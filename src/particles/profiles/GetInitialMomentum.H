#ifndef GETINTIALMOMENTUM_H_
#define GETINTIALMOMENTUM_H_

#include "particles/BeamParticleContainer.H"
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>


/** \brief Beam profile type, currently only Gaussian or Flattop */
enum struct BeamMomentumType { Gaussian, Flattop };

struct GetInitialMomentum
{
    GetInitialMomentum () = default;

    GetInitialMomentum (amrex::Real m_density); //, amrex::Real mean, amrex::Real std);

    void operator() (amrex::Real& ux, amrex::Real& uy, amrex::Real& uz ); //, amrex::Real x, amrex::Real y, amrex::Real z);
    amrex::RealVect  m_u_mean;
    amrex::RealVect  m_u_std {0.,0.,0.};
    BeamMomentumType m_momentum_profile;
};

#endif // GETINTIALMOMENTUM_H_
