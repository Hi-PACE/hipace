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

#include <AMReX.H>
#include <AMReX_REAL.H>

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
     * \param[in] w_index index to the weight component
     */
    template<class PTD>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PTD& ptd,
        const int ip, amrex::Real& x, amrex::Real& y,
        amrex::Real& ux, amrex::Real& uy, const int w_index) const noexcept
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
            } else {
                ptd.rdata(w_index)[ip] = amrex::Real(0);
                ptd.id(ip).make_invalid();
                invalid = true;
            }
        }

        return invalid;
    }
};

#endif // HIPACE_GETANDSETPOSITION_H_
