/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_GETANDSETPOSITION_H_
#define HIPACE_GETANDSETPOSITION_H_

#include "particles/plasma/PlasmaParticleContainer.H"

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <limits>


/** \brief Functor that can be used to extract the positions of the macroparticles
 *         inside a ParallelFor kernel
 */
template <class T_ParTile>
struct GetParticlePosition
{
    using PType = typename T_ParTile::ParticleType;
    using RType = amrex::ParticleReal;

    const PType* m_structs;

    /** Default constructor */
    GetParticlePosition () = default;

    /** Constructor.
     * \param a_ptile tile containing the macroparticles
     * \param a_offset offset to apply to the particle indices
     */
    GetParticlePosition (const T_ParTile& a_ptile, int a_offset = 0) noexcept
    {
        const auto& aos = a_ptile.GetArrayOfStructs();
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

    /** \brief Get the position of the particle at index `i + a_offset`, and put it in x, y and z
     * \param[in] i index of the particle
     * \param[in,out] x x position of particle i, modified by this function
     * \param[in,out] y y position of particle i, modified by this function
     * \param[in,out] z z position of particle i, modified by this function
     * \param[in,out] id id of particle i, modified by this function
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const int i, RType& x, RType& y, RType& z, int& id) const noexcept
    {
        x = m_structs[i].pos(0);
        y = m_structs[i].pos(1);
        z = m_structs[i].pos(2);
        id = m_structs[i].id();
    }
};

/** \brief Functor that can be used to modify the positions of the macroparticles
 *         inside a ParallelFor kernel
 */
template <class T_ParTile>
struct SetParticlePosition
{
    using PType = typename T_ParTile::ParticleType;
    using RType = amrex::ParticleReal;

    PType* m_structs;

    /** Constructor.
     * \param a_ptile tile containing the macroparticles
     * \param a_offset offset to apply to the particle indices
     */
    SetParticlePosition (T_ParTile& a_ptile, int a_offset = 0) noexcept
    {
        auto& aos = a_ptile.GetArrayOfStructs();
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

    /** \brief Set the position of the particle at index `i + a_offset` from values in x, y and z
     * \param[in] i index of the particle
     * \param[in] x new x position of particle i
     * \param[in] y new x position of particle i
     * \param[in] z new x position of particle i
     * \param[in] id new id of particle i
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const int i, RType x, RType y, RType z, int id) const noexcept
    {

        m_structs[i].pos(0) = x;
        m_structs[i].pos(1) = y;
        m_structs[i].pos(2) = z;
        m_structs[i].id() = id;

    }
};

/** \brief Functor that can be used to apply the boundary conditions to the macroparticles
 *         inside a ParallelFor kernel
 */
template <class T_ParTile>
struct EnforceBCandSetPos
{
    using PType = typename T_ParTile::ParticleType;
    using RType = amrex::ParticleReal;

    PType* m_structs;
    RType* m_weights;

    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_plo;
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_phi;
    int m_is_per;

    /** Constructor.
     * \param a_ptile tile containing the macroparticles
     * \param geom geometry used for domain size and periodicity
     * \param a_offset offset to apply to the particle indices
     */
    EnforceBCandSetPos (T_ParTile& a_ptile, const amrex::Geometry& geom, int a_offset = 0) noexcept
    {

        m_plo    = geom.ProbLoArray();
        m_phi    = geom.ProbHiArray();
        auto is_per = geom.isPeriodicArray();
        m_is_per = is_per[0];
        AMREX_ALWAYS_ASSERT(is_per[0] == is_per[1]);

        auto& aos = a_ptile.GetArrayOfStructs();
        m_structs = aos().dataPtr() + a_offset;
        auto& soa = a_ptile.GetStructOfArrays();
        m_weights = soa.GetRealData(BeamIdx::w).data() + a_offset;
    }

    /** \brief set position and enforces the boundary condition to the particle
     * at index `i + a_offset` and returns if the particle is invalid
     * \param[in] ip index of the particle
     * \param[in] x new x position of particle
     * \param[in] y new x position of particle
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const int ip, RType& x, RType& y) const noexcept
    {
        using namespace amrex::literals;

        m_structs[ip].pos(0) = x;
        m_structs[ip].pos(1) = y;

        // TODO: The second m_phi should be amrex::Geometry RoundoffHiArray(),
        // however there is no Geometry object to get this.
        const bool shifted = enforcePeriodic(m_structs[ip], m_plo, m_phi, m_plo, m_phi, {true, true, false});
        const bool invalid = (shifted && !m_is_per);
        if (invalid) {
            m_weights[ip] = 0.0_rt;
            m_structs[ip].id() = -std::abs(m_structs[ip].id());
        }
        x = m_structs[ip].pos(0);
        y = m_structs[ip].pos(1);
        return invalid;
    }

    /** \brief set position and enforces the boundary condition to the particle
     * at index `i + a_offset` and returns if the particle is invalid
     * \param[in] ip index of the particle
     * \param[in] x new x position of particle
     * \param[in] y new x position of particle
     * \param[in] z new x position of particle
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const int ip, RType& x, RType& y, RType& z) const noexcept
    {
        using namespace amrex::literals;

        m_structs[ip].pos(0) = x;
        m_structs[ip].pos(1) = y;
        m_structs[ip].pos(2) = z;

        // TODO: The second m_phi should be amrex::Geometry RoundoffHiArray(),
        // however there is no Geometry object to get this.
        const bool shifted = enforcePeriodic(m_structs[ip], m_plo, m_phi, m_plo, m_phi, {true, true, false});
        const bool invalid = (shifted && !m_is_per);
        if (invalid) {
            m_weights[ip] = 0.0_rt;
            m_structs[ip].id() = -std::abs(m_structs[ip].id());
        }
        x = m_structs[ip].pos(0);
        y = m_structs[ip].pos(1);
        z = m_structs[ip].pos(2);
        return invalid;
    }
};

#endif // HIPACE_GETANDSETPOSITION_H_
