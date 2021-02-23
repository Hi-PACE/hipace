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
 * \param[in] focusing_field_strength strength of the linear field: Ex = field_strength * x
 * \param[in] accel_field_strength strength of the linear field: Ez = field_strength * z
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ApplyExternalField(
    const amrex::ParticleReal xp,
    const amrex::ParticleReal yp,
    const amrex::ParticleReal zp,
    amrex::ParticleReal& ExmByp,
    amrex::ParticleReal& EypBxp,
    amrex::ParticleReal& Ezp,
    const amrex::Real focusing_field_strength,
    const amrex::Real accel_field_strength,
    const amrex::Real accel_field)
{
    if ((focusing_field_strength == 0.) && (accel_field_strength == 0.)) return;

    ExmByp += focusing_field_strength * xp;
    EypBxp += focusing_field_strength * yp;
    Ezp += accel_field + accel_field_strength * zp;

    return;
}

#endif // EXTERNALFIELDS_H_
