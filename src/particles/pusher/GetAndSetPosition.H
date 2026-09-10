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

#include "Hipace.H"
#include "particles/particles_utils/ParticleUtil.H"

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>

#include <limits>


/** \brief Functor that can be used to apply the boundary conditions to the macroparticles
 *         inside a ParallelFor kernel
 */
struct EnforceBC
{
    ParticleBoundary::type m_particle_boundary;
    amrex::GpuArray<amrex::Real, 2> m_plo;
    amrex::GpuArray<amrex::Real, 2> m_phi;

    /** Constructor.
     */
    EnforceBC () noexcept
    {
        m_particle_boundary = Hipace::m_boundary_particles;
        m_plo[0] = Hipace::m_boundary_particle_lo[0];
        m_plo[1] = Hipace::m_boundary_particle_lo[1];
        m_phi[0] = Hipace::m_boundary_particle_hi[0];
        m_phi[1] = Hipace::m_boundary_particle_hi[1];
    }

    /** \brief enforces the boundary condition to the particle
     * at index `ip` and returns if the particle is invalid
     * \param[in] ptd ParticleTileData
     * \param[in] ip index of the particle
     * \param[in] x x position of particle
     * \param[in] y y position of particle
     * \param[in] ux x momentum of particle
     * \param[in] uy y momentum of particle
     */
    template<class PTD>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PTD& ptd,
        const int ip, amrex::Real& x, amrex::Real& y,
        amrex::Real& ux, amrex::Real& uy) const noexcept
    {
        bool invalid = false;
        if (x < m_plo[0] || y < m_plo[1] || x > m_phi[0] || y > m_phi[1]) {
            const amrex::Real len_x = m_phi[0] - m_plo[0];
            const amrex::Real len_y = m_phi[1] - m_plo[1];
            if (m_particle_boundary == ParticleBoundary::Reflecting) {
                x = std::fmod(x - m_plo[0], 2*len_x);
                if (x < 0) x += 2*len_x;
                x += m_plo[0];
                if (x > m_phi[0]) {
                    x = 2*m_phi[0] - x;
                    ux = -ux;
                }
                y = std::fmod(y - m_plo[1], 2*len_y);
                if (y < 0) y += 2*len_y;
                y += m_plo[1];
                if (y > m_phi[1]) {
                    y = 2*m_phi[1] - y;
                    uy = -uy;
                }
                invalid = false;
            } else if (m_particle_boundary == ParticleBoundary::Periodic) {
                x = std::fmod(x - m_plo[0], len_x);
                if (x < 0) x += len_x;
                x += m_plo[0];
                y = std::fmod(y - m_plo[1], len_y);
                if (y < 0) y += len_y;
                y += m_plo[1];
                invalid = false;
            } else if (m_particle_boundary == ParticleBoundary::Thermal) {
                ptd.id(ip) = PlasmaID::to_be_thermalised;
            } else {                       // ParticleBoundary::Absorbing
                ptd.id(ip) = PlasmaID::invalid_at_boundary;
                invalid = true;
            }
        }
        return invalid;
    }

    /** \brief enforces Thermal boundary condition to the particle
     * at index `ip`
     * \param[inout] ptd ParticleTileData
     * \param[in] ip index of the particle
     * \param[inout] x x position of particle
     * \param[inout] y y position of particle
     * \param[inout] ux x momentum of particle
     * \param[inout] uy y momentum of particle
     * \param[inout] uz z momentum of particle
     * \param[inout] u_std momentum spread from thermal boundary velocity
     * \param[inout] engine amrex's random number generator
     */
    template<class PTD>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const PTD& ptd,
        const int ip, amrex::Real& x, amrex::Real& y,
        amrex::Real& ux, amrex::Real& uy, amrex::Real& uz, amrex::Real u_std,
        const amrex::RandomEngine& engine) const noexcept
    {
        // only called to reflect particle positions and provide the thermal
        // velocities to the particles

        // get a random thermal velocity vector
        amrex::Real u[3] = {0.,0.,0.};
        ParticleUtil::get_gaussian_random_momentum(u, {0., 0., 0.}, {u_std, u_std, u_std}, engine);
        ux = u[0];
        uy = u[1];
        uz = u[2];

        // use same method for reflecting as in the ParticleBoundary::Reflecting case
        const amrex::Real len_x = m_phi[0] - m_plo[0];
        const amrex::Real len_y = m_phi[1] - m_plo[1];

        if (x < m_plo[0] || x > m_phi[0]) { // reflect particle in x
            if (x < m_plo[0]) ux = (ux > 0) ? ux : -ux;
            else ux = (ux < 0) ? ux : -ux;

            x = std::fmod(x - m_plo[0], 2*len_x);
            if (x < 0) x += 2*len_x;
            x += m_plo[0];
            if (x > m_phi[0]) {
                x = 2*m_phi[0] - x;
            }
        }
        if (y < m_plo[1] || y > m_phi[1]) { // reflect particle in y
            if (y < m_plo[1]) uy = (uy > 0) ? uy : -uy;
            else uy = (uy < 0) ? uy : -uy;
            
            y = std::fmod(y - m_plo[1], 2*len_y);
            if (y < 0) y += 2*len_y;
            y += m_plo[1];
            if (y > m_phi[1]) {
                y = 2*m_phi[1] - y;
            }
        }
        // plasma particle is now valid again
        ptd.id(ip) = PlasmaID::valid;
    }
};

#endif // HIPACE_GETANDSETPOSITION_H_
