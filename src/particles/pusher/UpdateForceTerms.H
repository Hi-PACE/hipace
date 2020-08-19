#ifndef UPDATEFORCETERMS_H_
#define UPDATEFORCETERMS_H_


// /* TO DO: INLINE FUNCTION TO UPDATE THE FORCE TERMS OF A SINGLE PARTICLE */
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// void UpdateForceTerms(const amrex::ParticleReal& uxp,
//                       const amrex::ParticleReal& uyp,
//                       const amrex::ParticleReal& psip,
//                       const amrex::Real& gammap,
//                       const amrex::ParticleReal& ExmByp,
//                       const amrex::ParticleReal& EypBxp,
//                       const amrex::ParticleReal& Ezp,
//                       const amrex::ParticleReal& Bxp,
//                       const amrex::ParticleReal& Byp,
//                       const amrex::ParticleReal& Bzp,
//                       amrex::ParticleReal& Fx1,
//                       amrex::ParticleReal& Fy1,
//                       amrex::ParticleReal& Fux1,
//                       amrex::ParticleReal& Fuy1,
//                       amrex::ParticleReal& Fpsi1,
//                       const amrex::Real charge_mass_ratio)
//
// {
//   /* Change for x-position along zeta */
//   Fx1 = -uxp / psip;
//   /* Change for y-position along zeta */
//   Fy1 = -uyp / psip;
//   /* Change for ux along zeta */
//   Fux1 = -charge_mass_ratio * ( gammap * ExmByp / psip + Byp + ( uyp * Bzp ) / psip );
//   /* Change for uy along zeta */
//   Fuy1 = -charge_mass_ratio * ( gammap * EypBxp / psip - Bxp - ( uxp * Bzp ) / psip );
//   /* Change for psi along zeta */
//   Fpsi1 = -charge_mass_ratio * (( uxp * ExmByp + uyp * EypBxp ) / psip - Ezp );
// }

#endif //  UPDATEFORCETERMS_H_
