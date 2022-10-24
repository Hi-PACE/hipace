/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_BoxSort_H_
#define HIPACE_BoxSort_H_

#include "BeamParticleContainer.H"

#include <AMReX_MultiFab.H>

class BoxSorter
{
public:
    using index_type = unsigned int;

    void sortParticlesByBox (BeamParticleContainer& a_beam,
                             const amrex::BoxArray a_ba, const amrex::Geometry& a_geom);

    //! \brief returns the pointer to the permutation array
    index_type* boxCountsPtr () noexcept { return m_box_counts_cpu.dataPtr(); }

    //! \brief returns the pointer to the offsets array
    index_type* boxOffsetsPtr () noexcept { return m_box_offsets_cpu.dataPtr(); }

    //! \brief returns the pointer to the permutation array
    const index_type* boxCountsPtr () const noexcept { return m_box_counts_cpu.dataPtr(); }

    //! \brief returns the pointer to the offsets array
    const index_type* boxOffsetsPtr () const noexcept { return m_box_offsets_cpu.dataPtr(); }

    /** Get the index of the most downstream box that has beam particles */
    int leftmostBoxWithParticles () const;

private:
    /** Object to locate the box in which a particle is located */
    amrex::ParticleLocator<amrex::DenseBins<amrex::Box> > m_particle_locator;
    /** Number of particles in each box */
    amrex::Gpu::DeviceVector<index_type> m_box_counts;
    /** Number of particles in each box, stored on the cpu */
    amrex::Vector<index_type> m_box_counts_cpu;
    /** Index of the first particle in each box */
    amrex::Gpu::DeviceVector<index_type> m_box_offsets;
    /** Index of the first particle in each box, stored on the cpu*/
    amrex::Vector<index_type> m_box_offsets_cpu;
};

#endif // HIPACE_BoxSort_H_
