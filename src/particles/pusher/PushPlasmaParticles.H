#ifndef PUSHPLASMAPARTICLES_H_
#define PUSHPLASMAPARTICLES_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "FieldGather.H"
#include "Hipace.H"
#include "GetAndSetPosition.H"

/** \brief Pushing the plasma particles (currently only with a 5th order Adams Bashforth Pusher)
 *
 * \param[in,out] xp position in x direction
 * \param[in,out] yp position in y direction
 * \param[in,out] zp position in z direction
 * \param[in,out] uxp momentum in x direction
 * \param[in,out] uyp momentum in y direction
 * \param[in,out] psip plasma pseudo-potential
 * \param[in,out] x_temp temporary position in x direction
 * \param[in,out] y_temp temporary position in y direction
 * \param[in,out] ux_temp temporary momentum in x direction
 * \param[in,out] uy_temp temporary momentum in y direction
 * \param[in,out] psi_temp temporary plasma pseudo-potential
 * \param[in] Fx1 force term acting on x position of the particle
 * \param[in] Fy1 force term acting on y position of the particle
 * \param[in] Fux1 force term acting on x momentum of the particle
 * \param[in] Fuy1 force term acting on y momentum of the particle
 * \param[in] Fpsi1 force term acting on the plasma pseudo-potential of the particle
 * \param[in] Fx2 force term acting on x position of the particle
 * \param[in] Fy2 force term acting on y position of the particle
 * \param[in] Fux2 force term acting on x momentum of the particle
 * \param[in] Fuy2 force term acting on y momentum of the particle
 * \param[in] Fpsi2 force term acting on the plasma pseudo-potential of the particle
 * \param[in] Fx3 force term acting on x position of the particle
 * \param[in] Fy3 force term acting on y position of the particle
 * \param[in] Fux3 force term acting on x momentum of the particle
 * \param[in] Fuy3 force term acting on y momentum of the particle
 * \param[in] Fpsi3 force term acting on the plasma pseudo-potential of the particle
 * \param[in] Fx4 force term acting on x position of the particle
 * \param[in] Fy4 force term acting on y position of the particle
 * \param[in] Fux4 force term acting on x momentum of the particle
 * \param[in] Fuy4 force term acting on y momentum of the particle
 * \param[in] Fpsi4 force term acting on the plasma pseudo-potential of the particle
 * \param[in] Fx5 force term acting on x position of the particle
 * \param[in] Fy5 force term acting on y position of the particle
 * \param[in] Fux5 force term acting on x momentum of the particle
 * \param[in] Fuy5 force term acting on y momentum of the particle
 * \param[in] Fpsi5 force term acting on the plasma pseudo-potential of the particle
 * \param[in] dz longitudinal size of the cell
 * \param[in] which_slice defines if this or the next slice is handled
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) is used
 * \param[in] ip index of the plasma particle
 * \param[in] SetPosition Functor for setting the particle position
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PlasmaParticlePush (amrex::ParticleReal& xp,
                         amrex::ParticleReal& yp,
                         amrex::ParticleReal& zp,
                         amrex::ParticleReal& uxp,
                         amrex::ParticleReal& uyp,
                         amrex::ParticleReal& psip,
                         amrex::ParticleReal& x_temp,
                         amrex::ParticleReal& y_temp,
                         amrex::ParticleReal& ux_temp,
                         amrex::ParticleReal& uy_temp,
                         amrex::ParticleReal& psi_temp,
                         const amrex::ParticleReal& Fx1,
                         const amrex::ParticleReal& Fy1,
                         const amrex::ParticleReal& Fux1,
                         const amrex::ParticleReal& Fuy1,
                         const amrex::ParticleReal& Fpsi1,
                         const amrex::ParticleReal& Fx2,
                         const amrex::ParticleReal& Fy2,
                         const amrex::ParticleReal& Fux2,
                         const amrex::ParticleReal& Fuy2,
                         const amrex::ParticleReal& Fpsi2,
                         const amrex::ParticleReal& Fx3,
                         const amrex::ParticleReal& Fy3,
                         const amrex::ParticleReal& Fux3,
                         const amrex::ParticleReal& Fuy3,
                         const amrex::ParticleReal& Fpsi3,
                         const amrex::ParticleReal& Fx4,
                         const amrex::ParticleReal& Fy4,
                         const amrex::ParticleReal& Fux4,
                         const amrex::ParticleReal& Fuy4,
                         const amrex::ParticleReal& Fpsi4,
                         const amrex::ParticleReal& Fx5,
                         const amrex::ParticleReal& Fy5,
                         const amrex::ParticleReal& Fux5,
                         const amrex::ParticleReal& Fuy5,
                         const amrex::ParticleReal& Fpsi5,
                         const amrex::Real dz,
                         const WhichSlice which_slice,
                         const bool temp_slice,
                         const long ip,
                         const SetParticlePosition& SetPosition
                         )

{
    using namespace amrex::literals;
    /* Fifth order adams bashforth coefficients */
    const amrex::Real a1_times_dz = ( 1901._rt / 720._rt ) * dz;
    const amrex::Real a2_times_dz = ( -1387._rt / 360._rt ) * dz;
    const amrex::Real a3_times_dz = ( 109._rt / 30._rt ) * dz;
    const amrex::Real a4_times_dz = ( -637._rt / 360._rt ) * dz;
    const amrex::Real a5_times_dz = ( 251._rt / 720._rt ) * dz;

    if (!temp_slice)
    {
        xp -= a1_times_dz * Fx1 + a2_times_dz * Fx2 + a3_times_dz * Fx3
            + a4_times_dz * Fx4 + a5_times_dz * Fx5;
        yp -= a1_times_dz * Fy1 + a2_times_dz * Fy2 + a3_times_dz * Fy3
            + a4_times_dz * Fy4 + a5_times_dz * Fy5;

        SetPosition(ip, xp, yp, zp);

        uxp -= a1_times_dz * Fux1 + a2_times_dz * Fux2 + a3_times_dz * Fux3
            + a4_times_dz * Fux4 + a5_times_dz * Fux5;
        uyp -= a1_times_dz * Fuy1 + a2_times_dz * Fuy2 + a3_times_dz * Fuy3
            + a4_times_dz * Fuy4 + a5_times_dz * Fuy5;
        psip -= a1_times_dz * Fpsi1 + a2_times_dz * Fpsi2 + a3_times_dz * Fpsi3
            + a4_times_dz * Fpsi4 + a5_times_dz * Fpsi5;
    }
    else
    {
        x_temp = xp - ( a1_times_dz * Fx1 + a2_times_dz * Fx2 + a3_times_dz * Fx3
                 + a4_times_dz * Fx4 + a5_times_dz * Fx5 );
        y_temp = yp - ( a1_times_dz * Fy1 + a2_times_dz * Fy2 + a3_times_dz * Fy3
                 + a4_times_dz * Fy4 + a5_times_dz * Fy5 );
        ux_temp = uxp - ( a1_times_dz * Fux1 + a2_times_dz * Fux2 + a3_times_dz * Fux3
                  + a4_times_dz * Fux4 + a5_times_dz * Fux5 );
        uy_temp = uyp - ( a1_times_dz * Fuy1 + a2_times_dz * Fuy2 + a3_times_dz * Fuy3
                  + a4_times_dz * Fuy4 + a5_times_dz * Fuy5 );
        psi_temp = psip - ( a1_times_dz * Fpsi1 + a2_times_dz * Fpsi2 + a3_times_dz * Fpsi3
                  + a4_times_dz * Fpsi4 + a5_times_dz * Fpsi5 );
    }
}

#endif //  PUSHPLASMAPARTICLES_H_
