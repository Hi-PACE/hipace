/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_GETANDSETPOSITION_H_
#define HIPACE_GETANDSETPOSITION_H_

#include "particles/PlasmaParticleContainer.H"

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

    const PType* AMREX_RESTRICT m_structs;

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

    PType* AMREX_RESTRICT m_structs;

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
struct EnforceBC
{
    using PType = typename T_ParTile::ParticleType;
    using RType = amrex::ParticleReal;

    PType* AMREX_RESTRICT m_structs;
    RType* AMREX_RESTRICT m_weights;

    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_plo;
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_phi;
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_plo_lev1;
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_phi_lev1;
    amrex::GpuArray<int,AMREX_SPACEDIM> m_is_per;
    amrex::GpuArray<int,AMREX_SPACEDIM> m_periodicity;

    /** Constructor.
     * \param a_ptile tile containing the macroparticles
     * \param lev level of MR
     * \param box box to get the correct physical boundaries
     * \param a_offset offset to apply to the particle indices
     */
    EnforceBC (T_ParTile& a_ptile, const int lev, const amrex::Box box, int a_offset = 0) noexcept
    {

        m_plo    = Hipace::GetInstance().Geom(lev).ProbLoArray();
        m_phi    = Hipace::GetInstance().Geom(lev).ProbHiArray();
        m_is_per = Hipace::GetInstance().Geom(lev).isPeriodicArray();
        AMREX_ALWAYS_ASSERT(m_is_per[0] == m_is_per[1]);

        if (lev == 1) {
            const auto dx = Hipace::GetInstance().Geom(lev).CellSizeArray();
            const amrex::IntVect& small = box.smallEnd();
            const amrex::IntVect& big = box.bigEnd();

            const auto plo = m_plo;
            m_plo_lev1 = {{AMREX_D_DECL(plo[0]+(small[0]+0.5)*dx[0], plo[1]+(small[1]+0.5)*dx[1],
                                        plo[2]+(small[2]+0.5)*dx[2])}};
            m_phi_lev1 =  {{AMREX_D_DECL(plo[0]+(big[0]+0.5)*dx[0], plo[1]+(big[1]+0.5)*dx[1],
                                         plo[2]+(big[2]+0.5)*dx[2])}};
        // amrex::Print() << " m_plo_lev1 " << m_plo_lev1[0] << " " << m_plo_lev1[1] << " " << m_plo_lev1[2]  << " m_phi_lev1 " << m_phi_lev1[0] << " " << m_phi_lev1[1] << " "<< m_phi_lev1[2] << "\n";
        } else {
            amrex::ignore_unused(box);
        }

        m_periodicity = {true, true, false};

        auto& aos = a_ptile.GetArrayOfStructs();
        m_structs = aos().dataPtr() + a_offset;
        auto& soa = a_ptile.GetStructOfArrays();
        m_weights = soa.GetRealData(BeamIdx::w).data() + a_offset;
    }

    /** \brief enforces the boundary condition to the particle at index `i + a_offset`
     * and returns if the particle is invalid
     * \param[in] ip index of the particle
     * \param[in] lev level of mesh refinement
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const int ip, const int lev) const noexcept
    {
        using namespace amrex::literals;

        bool shifted;
        if (lev == 0) {
            shifted = enforcePeriodic(m_structs[ip], m_plo, m_phi, m_periodicity);
        } else {
            shifted = enforcePeriodic(m_structs[ip], m_plo_lev1, m_phi_lev1, m_periodicity);
        }

        const bool invalid = (shifted && !m_is_per[0]);
        if (invalid) {
            m_weights[ip] = 0.0_rt;
            m_structs[ip].id() = -std::abs(m_structs[ip].id());
        }
        return invalid;
    }
};

#endif // HIPACE_GETANDSETPOSITION_H_
