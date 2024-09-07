/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "MultiPlasma.H"
#include "particles/deposition/PlasmaDepositCurrent.H"
#include "particles/deposition/ExplicitDeposition.H"
#include "particles/pusher/PlasmaParticleAdvance.H"
#include "particles/sorting/TileSort.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/DeprecatedInput.H"
#include "utils/IOUtil.H"
#include "Hipace.H"

MultiPlasma::MultiPlasma ()
{
    amrex::ParmParse pp("plasmas");
    queryWithParser(pp, "names", m_names);
    queryWithParser(pp, "adaptive_density", m_adaptive_density);
    queryWithParser(pp, "sort_bin_size", m_sort_bin_size);

    DeprecatedInput("plasmas", "collisions",
                    "hipace.collisions", "", true);
    DeprecatedInput("plasmas", "background_density_SI",
                    "hipace.background_density_SI", "", true);

    if (m_names[0] == "no_plasma") return;
    m_nplasmas = m_names.size();
    for (int i = 0; i < m_nplasmas; ++i) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_names[i]!="beam", "Cannot have plasma with name 'beam'");
        m_all_plasmas.emplace_back(PlasmaParticleContainer(m_names[i]));
    }

}

void
MultiPlasma::InitData (amrex::Vector<amrex::BoxArray> slice_ba,
                       amrex::Vector<amrex::DistributionMapping> slice_dm,
                       amrex::Vector<amrex::Geometry> slice_gm, amrex::Vector<amrex::Geometry> gm)
{
    for (auto& plasma : m_all_plasmas) {
        // make it think there is only level 0
        plasma.SetParGDB(slice_gm[0], slice_dm[0], slice_ba[0]);
        plasma.InitData(gm[0]);

        if(plasma.m_can_ionize) {
            PlasmaParticleContainer* plasma_product = nullptr;
            for (int i=0; i<m_names.size(); ++i) {
                if(m_names[i] == plasma.m_product_name) {
                    plasma_product = &m_all_plasmas[i];
                }
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(plasma_product != nullptr,
                "Must specify a valid product plasma for ionization using ionization_product");
            plasma.InitIonizationModule(gm[0], plasma_product,
            Hipace::m_background_density_SI); // geometry only for dz
        }
    }
    if (m_nplasmas > 0) m_all_bins.resize(m_nplasmas);
}

amrex::Real
MultiPlasma::maxChargeDensity (amrex::Real z)
{
    amrex::Real max_density = std::abs(m_adaptive_density * get_phys_const().q_e);
    for (auto& plasma : m_all_plasmas) {
        plasma.UpdateDensityFunction(z);
        max_density = amrex::max<amrex::Real>(
            max_density, std::abs(plasma.GetCharge() * plasma.m_density_func(0., 0., z)));
    }
    return max_density;
}

void
MultiPlasma::DepositCurrent (
    Fields & fields, int which_slice,
    bool deposit_jx_jy, bool deposit_jz, bool deposit_rho, bool deposit_chi, bool deposit_rhomjz,
    amrex::Vector<amrex::Geometry> const& gm, int const lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        ::DepositCurrent(m_all_plasmas[i], fields, which_slice,
                         deposit_jx_jy, deposit_jz, deposit_rho, deposit_chi, deposit_rhomjz,
                         gm, lev, m_all_bins[i], m_sort_bin_size);
    }
}

void
MultiPlasma::ExplicitDeposition (Fields& fields, amrex::Vector<amrex::Geometry> const& gm,
                                 int const lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        ::ExplicitDeposition(m_all_plasmas[i], fields, gm, lev);
    }
}

void
MultiPlasma::AdvanceParticles (
    const Fields & fields, amrex::Vector<amrex::Geometry> const& gm, bool temp_slice, int lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        AdvancePlasmaParticles(m_all_plasmas[i], fields, gm, temp_slice, lev);
    }
}

void
MultiPlasma::DepositNeutralizingBackground (
    Fields & fields, int which_slice,
    amrex::Vector<amrex::Geometry> const& gm, int const lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        if (m_all_plasmas[i].m_neutralize_background) {
            // current of ions is zero, so they are not deposited.
            ::DepositCurrent(m_all_plasmas[i], fields, which_slice, false,
                             false, false, false, true, gm, lev, m_all_bins[i], m_sort_bin_size);
        }
    }
}

void
MultiPlasma::DoFieldIonization (
    const int lev, const amrex::Geometry& geom, const Fields& fields)
{
    for (auto& plasma : m_all_plasmas) {
        plasma.IonizationModule(lev, geom, fields, Hipace::m_background_density_SI);
    }
}

bool
MultiPlasma::IonizationOn () const
{
    bool ionization_on = false;
    for (auto& plasma : m_all_plasmas) {
        if (plasma.m_can_ionize) ionization_on = true;
    }
    return ionization_on;
}

bool
MultiPlasma::AnySpeciesNeutralizeBackground () const
{
    bool any_species_neutralize = false;
    for (auto& plasma : m_all_plasmas) {
        if (plasma.m_neutralize_background) any_species_neutralize = true;
    }
    return any_species_neutralize;
}

void
MultiPlasma::TileSort (amrex::Box bx, amrex::Geometry geom)
{
    m_all_bins.clear();
    for (auto& plasma : m_all_plasmas) {
        m_all_bins.emplace_back(
            findParticlesInEachTile(bx, m_sort_bin_size, plasma, geom));
    }
}

void
MultiPlasma::ReorderParticles (const int islice)
{
    for (auto& plasma : m_all_plasmas) {
        plasma.ReorderParticles(islice);
    }
}

void
MultiPlasma::TagByLevel (const int current_N_level, amrex::Vector<amrex::Geometry> const& geom3D,
                         const bool to_prev)
{
    for (auto& plasma : m_all_plasmas) {
        plasma.TagByLevel(current_N_level, geom3D, to_prev);
    }
}

void
MultiPlasma::InSituComputeDiags (int step, int islice, int max_step,
                                amrex::Real physical_time, amrex::Real max_time)
{
    for (auto& plasma : m_all_plasmas) {
        if (utils::doDiagnostics(plasma.m_insitu_period, step,
                            max_step, physical_time, max_time)) {
            plasma.InSituComputeDiags(islice);
        }
    }
}

void
MultiPlasma::InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom,
                                int max_step, amrex::Real max_time)
{
    for (auto& plasma : m_all_plasmas) {
        if (utils::doDiagnostics(plasma.m_insitu_period, step,
                            max_step, time, max_time)) {
            plasma.InSituWriteToFile(step, time, geom);
        }
    }
}
