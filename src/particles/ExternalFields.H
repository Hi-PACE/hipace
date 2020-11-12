#ifndef EXTERNALFIELDS_H_
#define EXTERNALFIELDS_H_

/** \brief add axisymmetric linear focusing field on particles. Typically called right after
 * the field gather.
 *
 * \param[in] xp particle position x
 * \param[in] yp particle position y
 * \param[in,out] ExmByp Ex-By Field on particle
 * \param[in,out] EypBxp Ey+Bx Field on particle
 * \param[in] field_strength strength of the linear field: Ex = field_strength * x
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ApplyExternalField(
    const amrex::ParticleReal xp,
    const amrex::ParticleReal yp,
    amrex::ParticleReal& ExmByp,
    amrex::ParticleReal& EypBxp,
    const amrex::Real field_strength)
{
    if (field_strength == 0.) return;

    ExmByp += field_strength * xp;
    EypBxp += field_strength * yp;

    return;
}

#endif // EXTERNALFIELDS_H_
