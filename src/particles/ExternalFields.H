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
 * \param[in] ext_ExmBy_strength strength of the external linear field: Ex += ext_ExmBy_strength * x
 * \param[in] ext_Ez_strength strength of the external linear field: Ez += ext_Ez_strength * z
 * \param[in] ext_Ez strength of the external uniform field: Ez += ext_Ez
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ApplyExternalField(
    const amrex::ParticleReal xp,
    const amrex::ParticleReal yp,
    const amrex::ParticleReal zp,
    amrex::ParticleReal& ExmByp,
    amrex::ParticleReal& EypBxp,
    amrex::ParticleReal& Ezp,
    const amrex::Real ext_ExmBy_strength,
    const amrex::Real ext_Ez_strength,
    const amrex::Real ext_Ez)
{
    if ((ext_ExmBy_strength == 0.) && (ext_Ez_strength == 0.)) return;

    ExmByp += ext_ExmBy_strength * xp;
    EypBxp += ext_ExmBy_strength * yp;
    Ezp += ext_Ez + ext_Ez_strength * zp;

    return;
}

#endif // EXTERNALFIELDS_H_
