#ifndef GETINTIALMOMENTUM_H_
#define GETINTIALMOMENTUM_H_

#include "particles/ParticleUtil.H"
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>

/** \brief Beam profile type, currently only Gaussian */
enum struct BeamMomentumType { Gaussian };

/** \brief Functor gets the initial momentum for beam particles
 */
struct GetInitialMomentum
{
    /** Constructor.
     */
    GetInitialMomentum (const std::string& name);

    /** \brief Get the momentum for a beam particle
     * \param[in,out] ux momentum in x, modified by this function
     * \param[in,out] uy momentum in y, modified by this function
     * \param[in,out] uz momentum in z, modified by this function
     * \param[in] engine random number engine
     * \param[in] z position in z
     * \param[in] duz_per_uz0_dzeta correlated energy spread
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (amrex::Real& ux, amrex::Real& uy, amrex::Real& uz,
                     const amrex::RandomEngine& engine, const amrex::Real z=0.,
                     const amrex::Real duz_per_uz0_dzeta=0.) const
    {
        amrex::Real u[3] = {ux,uy,uz};
        if (m_momentum_profile == BeamMomentumType::Gaussian){
            ParticleUtil::get_gaussian_random_momentum(u, m_u_mean, m_u_std, engine);
        }
        ux = u[0];
        uy = u[1];
        uz = u[2] + z*duz_per_uz0_dzeta*m_u_mean[2];
    }

    amrex::RealVect  m_u_mean;
    amrex::RealVect  m_u_std {0.,0.,0.};
    BeamMomentumType m_momentum_profile = BeamMomentumType::Gaussian;
};

#endif // GETINTIALMOMENTUM_H_
