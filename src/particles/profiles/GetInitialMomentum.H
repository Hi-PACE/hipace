#ifndef GETINTIALMOMENTUM_H_
#define GETINTIALMOMENTUM_H_

#include "particles/ParticleUtil.H"
// #include "particles/BeamParticleContainer.H"
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>


/** \brief Beam profile type, currently only Gaussian or Flattop */
enum struct BeamMomentumType { Gaussian, Flattop };

struct GetInitialMomentum
{
    // GetInitialMomentum () = default;

    GetInitialMomentum (); //, amrex::Real mean, amrex::Real std);

    void operator() (amrex::Real& ux, amrex::Real& uy, amrex::Real& uz ) const
    {
        amrex::Real u[3] = {ux,uy,uz};
        if (m_momentum_profile == BeamMomentumType::Gaussian){
            ParticleUtil::get_gaussian_random_momentum(u, m_u_mean, m_u_std);
        }
    }
    amrex::RealVect  m_u_mean;
    amrex::RealVect  m_u_std {0.,0.,0.};
    BeamMomentumType m_momentum_profile;
};

#endif // GETINTIALMOMENTUM_H_
