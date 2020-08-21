/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_GETANDSETPOSITION_H_
#define HIPACE_GETANDSETPOSITION_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include "particles/PlasmaParticleContainer.H"

#include <limits>


/** \brief Functor that can be used to extract the positions of the macroparticles
 *         inside a ParallelFor kernel
 */
struct GetParticlePosition
{
    using PType = PlasmaParticleContainer::ParticleType;
    using RType = amrex::ParticleReal;

    const PType* AMREX_RESTRICT m_structs;

    /** Default constructor */
    GetParticlePosition () = default;

    /** Constructor.
     * \param a_pti iterator to the tile containing the macroparticles
     * \param a_offset offset to apply to the particle indices
     */
    GetParticlePosition (const PlasmaParticleIterator& a_pti, int a_offset = 0) noexcept
    {
        const auto& aos = a_pti.GetArrayOfStructs();
        m_structs = aos().dataPtr() + a_offset;
    }

    /** \brief Get the position of the particle at index `i + a_offset`, and put it in x, y and z
     * \param[in] i index of the particle
     * \param[in,out] x x position of particle i, modified by this function
     * \param[in,out] y y position of particle i, modified by this function
     * \param[in,out] z z position of particle i, modified by this function
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const int i, RType& x, RType& y, RType& z) const noexcept
    {
        x = m_structs[i].pos(0);
        y = m_structs[i].pos(1);
        z = m_structs[i].pos(2);
    }
};

/** \brief Functor that can be used to modify the positions of the macroparticles
 *         inside a ParallelFor kernel
 */
struct SetParticlePosition
{
    using PType = PlasmaParticleContainer::ParticleType;
    using RType = amrex::ParticleReal;

    PType* AMREX_RESTRICT m_structs;

    /** Constructor.
     * \param a_pti iterator to the tile containing the macroparticles
     * \param a_offset offset to apply to the particle indices
     */
    SetParticlePosition (PlasmaParticleIterator& a_pti, int a_offset = 0) noexcept
    {
        auto& aos = a_pti.GetArrayOfStructs();
        m_structs = aos().dataPtr() + a_offset;
    }

    /** \brief Set the position of the particle at index `i + a_offset` from values in x, y and z
     * \param[in] i index of the particle
     * \param[in] x new x position of particle i
     * \param[in] y new x position of particle i
     * \param[in] z new x position of particle i
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const int i, RType x, RType y, RType z) const noexcept
    {

        m_structs[i].pos(0) = x;
        m_structs[i].pos(1) = y;
        m_structs[i].pos(2) = z;

    }
};

#endif // HIPACE_GETANDSETPOSITION_H_
