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
                      const amrex::Real clightsq)

{
  using namespace amrex::literals;

  const amrex::Real gammap = (1.0_rt + uxp*uxp*clightsq
                                     + uyp*uyp*clightsq
                                     + psip*psip)/(2.0_rt * psip );

  const amrex::Real charge_mass_ratio = -1.0_rt;

  /* Change for x-position along zeta */
  Fx1 = -uxp / psip;
  /* Change for y-position along zeta */
  Fy1 = -uyp / psip;
  /* Change for ux along zeta */
  Fux1 = -charge_mass_ratio * ( gammap * ExmByp / psip + Byp + ( uyp * Bzp ) / psip );
  /* Change for uy along zeta */
  Fuy1 = -charge_mass_ratio * ( gammap * EypBxp / psip - Bxp - ( uxp * Bzp ) / psip );
  /* Change for psi along zeta */
  Fpsi1 = -charge_mass_ratio * (( uxp * ExmByp + uyp * EypBxp ) / psip - Ezp );
}

#endif //  UPDATEFORCETERMS_H_
