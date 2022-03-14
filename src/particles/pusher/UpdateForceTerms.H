/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
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
 * \param[in] phys_const struct with all physical constant, depending on units system
 * \param[in] charge charge of the particles
 * \param[in] mass mass of the particles
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
                      const PhysConst& phys_const,
                      const amrex::Real charge,
                      const amrex::Real mass)

{
    using namespace amrex::literals;

    const amrex::Real gammap = (1.0_rt + uxp*uxp*clightsq
                                       + uyp*uyp*clightsq
                                       + psip*psip
        ) / (2.0_rt * psip );

    const amrex::Real charge_mass_ratio = charge / mass;
    /* Change for x-position along zeta */
    Fx1 = -uxp / psip / phys_const.c;
    /* Change for y-position along zeta */
    Fy1 = -uyp / psip / phys_const.c;
    /* Change for ux along zeta */
    Fux1 = -charge_mass_ratio/phys_const.c * (gammap * ExmByp / psip +
            phys_const.c * Byp + ( uyp * Bzp ) / psip );
    /* Change for uy along zeta */
    Fuy1 = -charge_mass_ratio/phys_const.c * ( gammap * EypBxp / psip -
            phys_const.c * Bxp - ( uxp * Bzp ) / psip );
    /* Change for psi along zeta */
    Fpsi1 = -charge_mass_ratio /phys_const.c/phys_const.c *
            (( uxp*ExmByp + uyp*EypBxp )/phys_const.c/psip - Ezp );
}

/** \brief Shifting the force term coefficients
 *
 * \param[in] soa struct of arrays with force terms
 */
template<class StructOfArrays>
void ShiftForceTerms (StructOfArrays& soa)
{
    auto& rd = soa.GetRealData();

    rd[PlasmaIdx::Fx5].swap(rd[PlasmaIdx::Fx4]);
    rd[PlasmaIdx::Fy5].swap(rd[PlasmaIdx::Fy4]);
    rd[PlasmaIdx::Fux5].swap(rd[PlasmaIdx::Fux4]);
    rd[PlasmaIdx::Fuy5].swap(rd[PlasmaIdx::Fuy4]);
    rd[PlasmaIdx::Fpsi5].swap(rd[PlasmaIdx::Fpsi4]);

    rd[PlasmaIdx::Fx4].swap(rd[PlasmaIdx::Fx3]);
    rd[PlasmaIdx::Fy4].swap(rd[PlasmaIdx::Fy3]);
    rd[PlasmaIdx::Fux4].swap(rd[PlasmaIdx::Fux3]);
    rd[PlasmaIdx::Fuy4].swap(rd[PlasmaIdx::Fuy3]);
    rd[PlasmaIdx::Fpsi4].swap(rd[PlasmaIdx::Fpsi3]);

    rd[PlasmaIdx::Fx3].swap(rd[PlasmaIdx::Fx2]);
    rd[PlasmaIdx::Fy3].swap(rd[PlasmaIdx::Fy2]);
    rd[PlasmaIdx::Fux3].swap(rd[PlasmaIdx::Fux2]);
    rd[PlasmaIdx::Fuy3].swap(rd[PlasmaIdx::Fuy2]);
    rd[PlasmaIdx::Fpsi3].swap(rd[PlasmaIdx::Fpsi2]);

    rd[PlasmaIdx::Fx2].swap(rd[PlasmaIdx::Fx1]);
    rd[PlasmaIdx::Fy2].swap(rd[PlasmaIdx::Fy1]);
    rd[PlasmaIdx::Fux2].swap(rd[PlasmaIdx::Fux1]);
    rd[PlasmaIdx::Fuy2].swap(rd[PlasmaIdx::Fuy1]);
    rd[PlasmaIdx::Fpsi2].swap(rd[PlasmaIdx::Fpsi1]);

}

#endif //  UPDATEFORCETERMS_H_
