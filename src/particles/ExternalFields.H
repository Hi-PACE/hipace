#ifndef EXTERNALFIELDS_H_
#define EXTERNALFIELDS_H_

/** \brief add axisymmetric linear focusing field on particles and linear accelerating field.
 * Typically called right after
 * the field gather.
 *
 * \param[in] xp particle position x
 * \param[in] yp particle position y
 * \param[in] zp particle position x
 * \param[in,out] ExmByp Ex-By Field on particle
 * \param[in,out] EypBxp Ey+Bx Field on particle
 * \param[in,out] Ezp Ez Field on particle
 * \param[in] ext_ExmBy_slope strength of the external linear field: Ex += ext_ExmBy_slope * x
 * \param[in] ext_Ez_slope strength of the external linear field: Ez += ext_Ez_slope * z
 * \param[in] ext_Ez_uniform strength of the external uniform field: Ez += ext_Ez_uniform
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ApplyExternalField(
    const amrex::ParticleReal xp,
    const amrex::ParticleReal yp,
    const amrex::ParticleReal zp,
    amrex::ParticleReal& ExmByp,
    amrex::ParticleReal& EypBxp,
    amrex::ParticleReal& Ezp,
    const amrex::Real ext_ExmBy_slope,
    const amrex::Real ext_Ez_slope,
    const amrex::Real ext_Ez_uniform)
{
    if ((ext_ExmBy_slope == 0.) && (ext_Ez_slope == 0.) && ext_Ez_uniform) return;

    ExmByp += ext_ExmBy_slope * xp;
    EypBxp += ext_ExmBy_slope * yp;
    Ezp += ext_Ez_uniform + ext_Ez_uniform * zp;

    return;
}

#endif // EXTERNALFIELDS_H_
