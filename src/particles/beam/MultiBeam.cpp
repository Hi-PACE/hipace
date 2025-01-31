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
#include "utils/IOUtil.H"
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
    const bool do_beam_jx_jy_deposition, const bool do_beam_jz_deposition,
    const bool do_beam_rhomjz_deposition, const int which_slice, const int which_beam_slice,
    const bool only_highest)

{
    for (int i=0; i<m_nbeams; i++) {
        const bool is_salame = m_all_beams[i].m_do_salame && (step == 0);
        if ( is_salame || (which_slice != WhichSlice::Salame) ) {
            ::DepositCurrentSlice(m_all_beams[i], fields, geom, lev,
                                  do_beam_jx_jy_deposition && !is_salame,
                                  do_beam_jz_deposition,
                                  do_beam_rhomjz_deposition && !is_salame,
                                  which_slice, which_beam_slice, only_highest);
        }
    }
}

void
MultiBeam::shiftSlippedParticles (const int slice, amrex::Geometry const& geom)
{
    for (int i=0; i<m_nbeams; i++) {
        ::shiftSlippedParticles(m_all_beams[i], slice, geom);
    }
}

void
MultiBeam::AdvanceBeamParticlesSlice (
    const Fields& fields, amrex::Vector<amrex::Geometry> const& gm, const int slice,
    int const current_N_level)
{
    for (int i=0; i<m_nbeams; i++) {
        ::AdvanceBeamParticlesSlice(m_all_beams[i], fields, gm, slice, current_N_level);
    }
}

void
MultiBeam::TagByLevel (
    const int current_N_level, amrex::Vector<amrex::Geometry> const& geom3D, const int which_slice)
{
    for (int i=0; i<m_nbeams; i++) {
        m_all_beams[i].TagByLevel(current_N_level, geom3D, which_slice);
    }
}

void
MultiBeam::InSituComputeDiags (int step, int islice,
                               int max_step, amrex::Real physical_time,
                               amrex::Real max_time)
{
    for (auto& beam : m_all_beams) {
        if (utils::doDiagnostics(beam.m_insitu_period, step, max_step, physical_time, max_time)) {
            beam.InSituComputeDiags(islice);
        }
    }
}

void
MultiBeam::InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom,
                              int max_step, amrex::Real max_time)
{
    for (auto& beam : m_all_beams) {
        if (utils::doDiagnostics(beam.m_insitu_period, step, max_step, time, max_time)) {
            beam.InSituWriteToFile(step, time, geom);
        }
    }
}

void
MultiBeam::ReorderParticles (int beam_slice, int step, amrex::Geometry& slice_geom)
{
    for (auto& beam : m_all_beams) {
        beam.ReorderParticles(beam_slice, step, slice_geom);
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

bool MultiBeam::isSalameNow (const int step)
{
    if (step != 0) return false;

    for (int i = 0; i < m_nbeams; ++i) {
        if (m_all_beams[i].m_do_salame) {
            if (m_all_beams[i].getNumParticles(WhichBeamSlice::This) > 0) {
                return true;
            }
        }
    }
    return false;
}
