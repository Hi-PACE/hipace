/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_BoxSort_H_
#define HIPACE_BoxSort_H_

#include <AMReX_MultiFab.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>

class BeamParticleContainer;

class BoxSorter
{
public:
    using index_type = unsigned long long;

    void sortParticlesByBox (const amrex::Real * z_array,
                             const index_type num_particles,
                             const bool init_on_cpu,
                             const amrex::Geometry& a_geom);

    //! \brief returns the pointer to the permutation array
    index_type* boxCountsPtr () noexcept { return m_box_counts_cpu.dataPtr(); }

    //! \brief returns the pointer to the offsets array
    index_type* boxOffsetsPtr () noexcept { return m_box_offsets_cpu.dataPtr(); }

    //! \brief returns the pointer to the permutation array
    const index_type* boxCountsPtr () const noexcept { return m_box_counts_cpu.dataPtr(); }

    //! \brief returns the pointer to the offsets array
    const index_type* boxOffsetsPtr () const noexcept { return m_box_offsets_cpu.dataPtr(); }


    /** Number of particles in each box, stored on the cpu */
    amrex::Vector<index_type> m_box_counts_cpu;
    /** Index of the first particle in each box, stored on the cpu*/
    amrex::Vector<index_type> m_box_offsets_cpu;
    /** Index permutations m_box_permutations[new_idx] == old_idx
     * use PolymorphicArenaAllocator to either use Pinned or Device memory at runtime */
    amrex::PODVector<index_type, amrex::PolymorphicArenaAllocator<index_type>> m_box_permutations;
};

#endif // HIPACE_BoxSort_H_
