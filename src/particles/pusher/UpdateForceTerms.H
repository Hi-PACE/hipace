#ifndef UPDATEFORCETERMS_H_
#define UPDATEFORCETERMS_H_



/** \brief updating the force terms on a single plasma particle
 *
 * \param[in] uxp momentum in x direction
 * \param[in] uyp momentum in y direction
 * \param[in] psip plasma pseudo-potential
 * \param[in] ExmByp ExmBy field at particle position
 * \param[in] EypBxp EypBx field at particle position
 * \param[in] Ezp Ez field at particle position
 * \param[in] Bxp Bx field at particle position
 * \param[in] Byp By field at particle position
 * \param[in] Bzp Bz field at particle position
 * \param[in,out] Fx1 force term acting on x position of the particle
 * \param[in,out] Fy1 force term acting on y position of the particle
 * \param[in,out] Fux1 force term acting on x momentum of the particle
 * \param[in,out] Fuy1 force term acting on y momentum of the particle
 * \param[in,out] Fpsi1 force term acting on the plasma pseudo-potential of the particle
 * \param[in] clightsq 1/c0^2
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void UpdateForceTerms(const amrex::ParticleReal& uxp,
                      const amrex::ParticleReal& uyp,
                      const amrex::ParticleReal& psip,
                      const amrex::ParticleReal& ExmByp,
                      const amrex::ParticleReal& EypBxp,
                      const amrex::ParticleReal& Ezp,
                      const amrex::ParticleReal& Bxp,
                      const amrex::ParticleReal& Byp,
                      const amrex::ParticleReal& Bzp,
                      amrex::ParticleReal& Fx1,
                      amrex::ParticleReal& Fy1,
                      amrex::ParticleReal& Fux1,
                      amrex::ParticleReal& Fuy1,
                      amrex::ParticleReal& Fpsi1,
                      const amrex::Real clightsq,
                      const PhysConst& phys_const)

{
    using namespace amrex::literals;

    const amrex::Real gammap = (1.0_rt + uxp*uxp*clightsq
                                       + uyp*uyp*clightsq
                                       + (psip+1.0_rt)*(psip+1.0_rt))/(2.0_rt * (psip + 1.0_rt) );

    const amrex::Real charge_mass_ratio = -1.0_rt;
    
    /* Change for x-position along zeta */
    Fx1 = -uxp / (psip + 1.0_rt) / phys_const.c;
    /* Change for y-position along zeta */
    Fy1 = -uyp / (psip + 1.0_rt) / phys_const.c;
    /* Change for ux along zeta */
    Fux1 = - phys_const.q_e / phys_const.c / phys_const.m_e * charge_mass_ratio * ( gammap * ExmByp / (psip + 1.0_rt) + phys_const.c * Byp + ( uyp * Bzp )
            / (psip + 1.0_rt) );
    /* Change for uy along zeta */
    Fuy1 = -phys_const.q_e / phys_const.c / phys_const.m_e * charge_mass_ratio * ( gammap * EypBxp / (psip + 1.0_rt) - phys_const.c * Bxp - ( uxp * Bzp )
            / (psip + 1.0_rt) );
    /* Change for psi along zeta */
    Fpsi1 = -charge_mass_ratio * (1. / phys_const.c * ( uxp * ExmByp + uyp * EypBxp ) / (psip + 1.0_rt) - Ezp );

}

/** \brief Shifting the force term coefficients
 *
 * \param[in] Fx1 force term acting on x position of the particle
 * \param[in] Fy1 force term acting on y position of the particle
 * \param[in] Fux1 force term acting on x momentum of the particle
 * \param[in] Fuy1 force term acting on y momentum of the particle
 * \param[in] Fpsi1 force term acting on the plasma pseudo-potential of the particle
 * \param[in,out] Fx2 force term acting on x position of the particle
 * \param[in,out] Fy2 force term acting on y position of the particle
 * \param[in,out] Fux2 force term acting on x momentum of the particle
 * \param[in,out] Fuy2 force term acting on y momentum of the particle
 * \param[in,out] Fpsi2 force term acting on the plasma pseudo-potential of the particle
 * \param[in,out] Fx3 force term acting on x position of the particle
 * \param[in,out] Fy3 force term acting on y position of the particle
 * \param[in,out] Fux3 force term acting on x momentum of the particle
 * \param[in,out] Fuy3 force term acting on y momentum of the particle
 * \param[in,out] Fpsi3 force term acting on the plasma pseudo-potential of the particle
 * \param[in,out] Fx4 force term acting on x position of the particle
 * \param[in,out] Fy4 force term acting on y position of the particle
 * \param[in,out] Fux4 force term acting on x momentum of the particle
 * \param[in,out] Fuy4 force term acting on y momentum of the particle
 * \param[in,out] Fpsi4 force term acting on the plasma pseudo-potential of the particle
 * \param[in,out] Fx5 force term acting on x position of the particle
 * \param[in,out] Fy5 force term acting on y position of the particle
 * \param[in,out] Fux5 force term acting on x momentum of the particle
 * \param[in,out] Fuy5 force term acting on y momentum of the particle
 * \param[in,out] Fpsi5 force term acting on the plasma pseudo-potential of the particle
  */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ShiftForceTerms (const amrex::ParticleReal& Fx1,
                      const amrex::ParticleReal& Fy1,
                      const amrex::ParticleReal& Fux1,
                      const amrex::ParticleReal& Fuy1,
                      const amrex::ParticleReal& Fpsi1,
                      amrex::ParticleReal& Fx2,
                      amrex::ParticleReal& Fy2,
                      amrex::ParticleReal& Fux2,
                      amrex::ParticleReal& Fuy2,
                      amrex::ParticleReal& Fpsi2,
                      amrex::ParticleReal& Fx3,
                      amrex::ParticleReal& Fy3,
                      amrex::ParticleReal& Fux3,
                      amrex::ParticleReal& Fuy3,
                      amrex::ParticleReal& Fpsi3,
                      amrex::ParticleReal& Fx4,
                      amrex::ParticleReal& Fy4,
                      amrex::ParticleReal& Fux4,
                      amrex::ParticleReal& Fuy4,
                      amrex::ParticleReal& Fpsi4,
                      amrex::ParticleReal& Fx5,
                      amrex::ParticleReal& Fy5,
                      amrex::ParticleReal& Fux5,
                      amrex::ParticleReal& Fuy5,
                      amrex::ParticleReal& Fpsi5 )

{

    Fx5 = Fx4;
    Fy5 = Fy4;
    Fux5 = Fux4;
    Fuy5 = Fuy4;
    Fpsi5 = Fpsi4;

    Fx4 = Fx3;
    Fy4 = Fy3;
    Fux4 = Fux3;
    Fuy4 = Fuy3;
    Fpsi4 = Fpsi3;

    Fx3 = Fx2;
    Fy3 = Fy2;
    Fux3 = Fux2;
    Fuy3 = Fuy2;
    Fpsi3 = Fpsi2;

    Fx2 = Fx1;
    Fy2 = Fy1;
    Fux2 = Fux1;
    Fuy2 = Fuy1;
    Fpsi2 = Fpsi1;

}

#endif //  UPDATEFORCETERMS_H_
