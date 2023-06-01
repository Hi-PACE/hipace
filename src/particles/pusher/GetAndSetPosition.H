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


/** \brief Functor that can be used to apply the boundary conditions to the macroparticles
 *         inside a ParallelFor kernel
 */
template <class T_ParTile>
struct EnforceBCandSetPos
{
    using RType = amrex::ParticleReal;
    using PTDType = typename T_ParTile::ParticleTileDataType;

    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_plo;
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_phi;
    int m_is_per;

    /** Constructor.
     * \param geom geometry used for domain size and periodicity
     */
    EnforceBCandSetPos (const amrex::Geometry& geom) noexcept
    {

        m_plo    = geom.ProbLoArray();
        m_phi    = geom.ProbHiArray();
        auto is_per = geom.isPeriodicArray();
        m_is_per = is_per[0];
        AMREX_ALWAYS_ASSERT(is_per[0] == is_per[1]);
    }

    /** \brief set position and enforces the boundary condition to the plasma particle
     * at index `ip` and returns if the particle is invalid
     * \param[in] ptd ParticleTileData
     * \param[in] ip index of the particle
     * \param[in] x new x position of particle
     * \param[in] y new x position of particle
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PTDType& ptd, const int ip, RType& x, RType& y) const noexcept
    {
        using namespace amrex::literals;

        auto p = ptd[ip];

        p.pos(0) = x;
        p.pos(1) = y;

        // TODO: The second m_phi should be amrex::Geometry RoundoffHiArray(),
        // however there is no Geometry object to get this.
        const bool shifted = enforcePeriodic(p, m_plo, m_phi, m_plo, m_phi, {true, true, false});
        const bool invalid = (shifted && !m_is_per);
        if (invalid) {
            ptd.rdata(PlasmaIdx::w)[ip] = 0.0_rt;
            p.id() = -std::abs(p.id());
        }
        x = p.pos(0);
        y = p.pos(1);
        return invalid;
    }

    /** \brief set position and enforces the boundary condition to the beam particle
     * \param[in] ptd ParticleTileData
     * \param[in] ip index of the particle
     * \param[in] x new x position of particle
     * \param[in] y new x position of particle
     * \param[in] z new x position of particle
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PTDType& ptd, const int ip, RType& x, RType& y, RType& z) const noexcept
    {
        using namespace amrex::literals;

        auto p = ptd[ip];

        p.pos(0) = x;
        p.pos(1) = y;
        p.pos(2) = z;

        // TODO: The second m_phi should be amrex::Geometry RoundoffHiArray(),
        // however there is no Geometry object to get this.
        const bool shifted = enforcePeriodic(p, m_plo, m_phi, m_plo, m_phi, {true, true, false});
        const bool invalid = (shifted && !m_is_per);
        if (invalid) {
            ptd.rdata(BeamIdx::w)[ip] = 0.0_rt;
            p.id() = -std::abs(p.id());
        }
        x = p.pos(0);
        y = p.pos(1);
        z = p.pos(2);
        return invalid;
    }
};

#endif // HIPACE_GETANDSETPOSITION_H_
