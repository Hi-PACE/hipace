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
    using RType = amrex::ParticleReal;

    const RType * pos_x;
    const RType * pos_y;
    const RType * pos_z;
    const int * idp;

    /** Default constructor */
    GetParticlePosition () = default;

    /** Constructor.
     * \param a_ptile tile containing the macroparticles
     * \param a_offset offset to apply to the particle indices
     */
    GetParticlePosition (const T_ParTile& a_ptile, int a_offset = 0) noexcept
    {
        const auto& soa = a_ptile.GetStructOfArrays();
        pos_x = soa.GetRealData(0).data() + a_offset;
        pos_y = soa.GetRealData(1).data() + a_offset;
        pos_z = soa.GetRealData(2).data() + a_offset;
        idp = soa.GetIntData(0).data() + a_offset;
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
        x = pos_x[i];
        y = pos_y[i];
        z = pos_z[i];
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
        x = pos_x[i];
        y = pos_y[i];
        z = pos_z[i];
        id = idp[i];
    }
};

/** \brief Functor that can be used to modify the positions of the macroparticles
 *         inside a ParallelFor kernel
 */
template <class T_ParTile>
struct SetParticlePosition
{
    using RType = amrex::ParticleReal;

    RType * pos_x;
    RType * pos_y;
    RType * pos_z;
    int * idp;

    /** Constructor.
     * \param a_ptile tile containing the macroparticles
     * \param a_offset offset to apply to the particle indices
     */
    SetParticlePosition (T_ParTile& a_ptile, int a_offset = 0) noexcept
    {
        auto& soa = a_ptile.GetStructOfArrays();
        pos_x = soa.GetRealData(0).data() + a_offset;
        pos_y = soa.GetRealData(1).data() + a_offset;
        pos_z = soa.GetRealData(2).data() + a_offset;
        idp = soa.GetIntData(0).data() + a_offset;
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

        pos_x[i] = x;
        pos_y[i] = y;
        pos_z[i] = z;

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

        pos_x[i] = x;
        pos_y[i] = y;
        pos_z[i] = z;
        idp[i] = id;

    }
};

/** \brief Functor that can be used to apply the boundary conditions to the macroparticles
 *         inside a ParallelFor kernel
 */
template <class T_ParTile>
struct EnforceBCandSetPos
{
    using RType = amrex::ParticleReal;

    typename T_ParTile::ParticleTileDataType m_pardata;
    int m_offset;
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

        m_pardata = a_ptile.getParticleTileData();
        m_offset = a_offset;
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

        auto p = make_particle<typename T_ParTile::ParticleType>{}(m_pardata, ip + m_offset);

        p.pos(0) = x;
        p.pos(1) = y;

        // TODO: The second m_phi should be amrex::Geometry RoundoffHiArray(),
        // however there is no Geometry object to get this.
        const bool shifted = enforcePeriodic(p, m_plo, m_phi, m_plo, m_phi, {true, true, false});
        const bool invalid = (shifted && !m_is_per);
        if (invalid) {
            m_pardata.rdata(PlasmaIdx::w)[ip + m_offset] = 0.0_rt;
            p.id() = -std::abs(p.id());
        }
        x = p.pos(0);
        y = p.pos(1);
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

        auto p = make_particle<typename T_ParTile::ParticleType>{}(m_pardata, ip + m_offset);

        p.pos(0) = x;
        p.pos(1) = y;
        p.pos(2) = z;

        // TODO: The second m_phi should be amrex::Geometry RoundoffHiArray(),
        // however there is no Geometry object to get this.
        const bool shifted = enforcePeriodic(p, m_plo, m_phi, m_plo, m_phi, {true, true, false});
        const bool invalid = (shifted && !m_is_per);
        if (invalid) {
            m_pardata.rdata(BeamIdx::w)[ip + m_offset] = 0.0_rt;
            p.id() = -std::abs(p.id());
        }
        x = p.pos(0);
        y = p.pos(1);
        z = p.pos(2);
        return invalid;
    }
};

#endif // HIPACE_GETANDSETPOSITION_H_
