/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "MultiBeam.H"
#include "particles/deposition/BeamDepositCurrent.H"
#include "particles/sorting/SliceSort.H"
#include "particles/pusher/BeamParticleAdvance.H"
#include "utils/DeprecatedInput.H"
#include "utils/HipaceProfilerWrapper.H"

MultiBeam::MultiBeam ()
{
    amrex::ParmParse pp("beams");
    queryWithParser(pp, "names", m_names);
    if (m_names[0] == "no_beam") return;
    DeprecatedInput("beams", "insitu_freq", "insitu_period");
    DeprecatedInput("beams", "all_from_file",
        "injection_type = from_file\nand beams.input_file = <file name>\n");
    m_nbeams = m_names.size();
    for (int i = 0; i < m_nbeams; ++i) {
        m_all_beams.emplace_back(BeamParticleContainer(m_names[i]));
    }
    m_n_real_particles.resize(m_nbeams, 0);
}

amrex::Real
MultiBeam::InitData (const amrex::Geometry& geom)
{
    amrex::Real ptime {0.};
    for (auto& beam : m_all_beams) {
        ptime = beam.InitData(geom);
    }
    return ptime;
}

void
MultiBeam::DepositCurrentSlice (
    Fields& fields, amrex::Vector<amrex::Geometry> const& geom, const int lev, const int step,
    int islice, const bool do_beam_jx_jy_deposition, const bool do_beam_jz_deposition,
    const bool do_beam_rho_deposition, const int which_slice)

{
    for (int i=0; i<m_nbeams; i++) {
        const bool is_salame = m_all_beams[i].m_do_salame && (step == 0);
        if ( is_salame || (which_slice != WhichSlice::Salame) ) {
            const int nghost = m_all_beams[i].numParticles() - m_n_real_particles[i];
            ::DepositCurrentSlice(m_all_beams[i], fields, geom, lev, islice,
                                  do_beam_jx_jy_deposition && !is_salame,
                                  do_beam_jz_deposition,
                                  do_beam_rho_deposition && !is_salame,
                                  which_slice, nghost);
        }
    }
}

void
MultiBeam::findParticlesInEachSlice (int ibox, amrex::Box bx, amrex::Geometry const& geom)
{
    for (int i=0; i<m_nbeams; i++) {
        ::findParticlesInEachSlice(ibox, bx, m_all_beams[i], geom);
    }
    amrex::Gpu::streamSynchronize();
}

void
MultiBeam::sortParticlesByBox (const amrex::BoxArray a_ba, const amrex::Geometry& a_geom)
{
    for (int i=0; i<m_nbeams; i++) {
        m_all_beams[i].m_box_sorter.sortParticlesByBox(m_all_beams[i], a_ba, a_geom);
    }
}

void
MultiBeam::AdvanceBeamParticlesSlice (
    const Fields& fields, amrex::Geometry const& gm, int const lev, const int islice)
{
    for (int i=0; i<m_nbeams; i++) {
        ::AdvanceBeamParticlesSlice(m_all_beams[i], fields, gm, lev, islice);
    }
}

int
MultiBeam::NumRealComps ()
{
    int comps = 0;
    if (get_nbeams() > 0){
        comps = m_all_beams[0].NumRealComps();
        for (auto& beam : m_all_beams) {
            AMREX_ALWAYS_ASSERT(beam.NumRealComps() == comps);
        }
    }
    return comps;
}

int
MultiBeam::NumIntComps ()
{
    int comps = 0;
    if (get_nbeams() > 0){
        comps = m_all_beams[0].NumIntComps();
        for (auto& beam : m_all_beams) {
            AMREX_ALWAYS_ASSERT(beam.NumIntComps() == comps);
        }
    }
    return comps;
}

void
MultiBeam::StoreNRealParticles ()
{
    for (int i=0; i<m_nbeams; i++) {
        m_n_real_particles[i] = m_all_beams[i].numParticles();
    }
}

int
MultiBeam::NGhostParticles (int ibeam, amrex::Box bx)
{
    BeamBins::index_type const * offsets = 0;
    offsets = m_all_beams[ibeam].m_slice_bins.offsetsPtrCpu();
    return offsets[bx.bigEnd(Direction::z)+1-bx.smallEnd(Direction::z)]
        - offsets[bx.bigEnd(Direction::z)-bx.smallEnd(Direction::z)];
}

void
MultiBeam::RemoveGhosts ()
{
    for (int i=0; i<m_nbeams; i++){
        m_all_beams[i].resize(m_n_real_particles[i]);
    }
}

void
MultiBeam::SetIbox (const int ibox)
{
    for (int i=0; i<m_nbeams; i++){
        m_all_beams[i].m_ibox = ibox;
    }
}

void
MultiBeam::PackLocalGhostParticles (int it)
{
    HIPACE_PROFILE("MultiBeam::PackLocalGhostParticles()");
    for (int ibeam=0; ibeam<m_nbeams; ibeam++){

        const int offset_box_left = m_all_beams[ibeam].m_box_sorter.boxOffsetsPtr()[it];
        const int offset_box_curr = m_all_beams[ibeam].m_box_sorter.boxOffsetsPtr()[it+1];
        const int nghost = offset_box_curr - offset_box_left;

        // Resize particle array
        auto& ptile = getBeam(ibeam);
        int old_size = ptile.numParticles();
        auto new_size = old_size + nghost;
        ptile.resize(new_size);

        // Copy particles in box it to ghost particles
        // Access AoS particle data
        auto& aos = ptile.GetArrayOfStructs();
        const auto& pos_structs_src = aos.begin() + offset_box_left;
        const auto& pos_structs_dst = aos.begin() + old_size;
        // Access SoA particle data
        auto& soa = ptile.GetStructOfArrays(); // For momenta and weights
        const auto  wp_src = soa.GetRealData(BeamIdx::w).data()  + offset_box_left;
        const auto uxp_src = soa.GetRealData(BeamIdx::ux).data() + offset_box_left;
        const auto uyp_src = soa.GetRealData(BeamIdx::uy).data() + offset_box_left;
        const auto uzp_src = soa.GetRealData(BeamIdx::uz).data() + offset_box_left;
        const auto  wp_dst = soa.GetRealData(BeamIdx::w).data()  + old_size;
        const auto uxp_dst = soa.GetRealData(BeamIdx::ux).data() + old_size;
        const auto uyp_dst = soa.GetRealData(BeamIdx::uy).data() + old_size;
        const auto uzp_dst = soa.GetRealData(BeamIdx::uz).data() + old_size;

        amrex::ParallelFor(
            nghost,
            [=] AMREX_GPU_DEVICE (long idx) {
                pos_structs_dst[idx].id() = pos_structs_src[idx].id();
                pos_structs_dst[idx].pos(0) = pos_structs_src[idx].pos(0);
                pos_structs_dst[idx].pos(1) = pos_structs_src[idx].pos(1);
                pos_structs_dst[idx].pos(2) = pos_structs_src[idx].pos(2);
                wp_dst[idx] = wp_src[idx];
                uxp_dst[idx] = uxp_src[idx];
                uyp_dst[idx] = uyp_src[idx];
                uzp_dst[idx] = uzp_src[idx];
            }
            );
    }
}

void
MultiBeam::InSituComputeDiags (int step, int islice, int islice_local)
{
    for (int i = 0; i < m_nbeams; ++i) {
        if (m_all_beams[i].doInSitu(step)) {
            m_all_beams[i].InSituComputeDiags(islice, islice_local);
        }
    }
}

void
MultiBeam::InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom)
{
    for (auto& beam : m_all_beams) {
        if (beam.doInSitu(step)) {
            beam.InSituWriteToFile(step, time, geom);
        }
    }
}

bool MultiBeam::AnySpeciesSalame () {
    for (int i = 0; i < m_nbeams; ++i) {
        if (m_all_beams[i].m_do_salame) {
            return true;
        }
    }
    return false;
}

bool MultiBeam::isSalameNow (const int step, const int islice)
{
    if (step != 0) return false;

    for (int i = 0; i < m_nbeams; ++i) {
        if (m_all_beams[i].m_do_salame) {
            BeamBins::index_type const * const offsets =m_all_beams[i].m_slice_bins.offsetsPtrCpu();
            if ((offsets[islice + 1] - offsets[islice]) > 0) {
                return true;
            }
        }
    }
    return false;
}
