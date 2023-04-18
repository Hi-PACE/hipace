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
#include "Hipace.H"

MultiPlasma::MultiPlasma ()
{
    amrex::ParmParse pp("plasmas");
    queryWithParser(pp, "names", m_names);
    queryWithParser(pp, "adaptive_density", m_adaptive_density);
    queryWithParser(pp, "sort_bin_size", m_sort_bin_size);
    queryWithParser(pp, "collisions", m_collision_names);

    if (m_names[0] == "no_plasma") return;
    m_nplasmas = m_names.size();
    for (int i = 0; i < m_nplasmas; ++i) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_names[i]!="beam", "Cannot have plasma with name 'beam'");
        m_all_plasmas.emplace_back(PlasmaParticleContainer(m_names[i]));
    }

    /** Initialize the collision objects */
    m_ncollisions = m_collision_names.size();
     for (int i = 0; i < m_ncollisions; ++i) {
         m_all_collisions.emplace_back(CoulombCollision(m_names, m_collision_names[i]));
     }
     if (m_ncollisions > 0) {
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
             Hipace::m_normalized_units == false,
             "Coulomb collisions only work with normalized units for now");
     }
}

void
MultiPlasma::InitData (amrex::Vector<amrex::BoxArray> slice_ba,
                       amrex::Vector<amrex::DistributionMapping> slice_dm,
                       amrex::Vector<amrex::Geometry> slice_gm, amrex::Vector<amrex::Geometry> gm)
{
    HIPACE_PROFILE("MultiPlasma::InitData()");
    for (auto& plasma : m_all_plasmas) {
        // make it think there is only level 0
        plasma.SetParGDB(slice_gm[0], slice_dm[0], slice_ba[0]);
        plasma.InitData();

        if(plasma.m_can_ionize) {
            PlasmaParticleContainer* plasma_product = nullptr;
            for (int i=0; i<m_names.size(); ++i) {
                if(m_names[i] == plasma.m_product_name) {
                    plasma_product = &m_all_plasmas[i];
                }
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(plasma_product != nullptr,
                "Must specify a valid product plasma for Ionization using ionization_product");
            plasma.InitIonizationModule(gm[0], plasma_product); // geometry only for dz
        }
    }
    if (m_nplasmas > 0) m_all_bins.resize(m_nplasmas);
}

amrex::Real
MultiPlasma::maxDensity (amrex::Real z) const
{
    amrex::Real max_density = 0;
    for (auto& plasma : m_all_plasmas) {
        max_density = amrex::max<amrex::Real>(max_density, plasma.m_density_func(0., 0., z));
    }
    return amrex::max(max_density, m_adaptive_density);
}

void
MultiPlasma::DepositCurrent (
    Fields & fields, const MultiLaser & multi_laser, int which_slice,
    bool deposit_jx_jy, bool deposit_jz, bool deposit_rho, bool deposit_chi,
    amrex::Geometry const& gm, int const lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        ::DepositCurrent(m_all_plasmas[i], fields, multi_laser, which_slice,
                         deposit_jx_jy, deposit_jz, deposit_rho, deposit_chi,
                         gm, lev, m_all_bins[i], m_sort_bin_size);
    }
}

void
MultiPlasma::ExplicitDeposition (Fields& fields, const MultiLaser& multi_laser,
                                 amrex::Geometry const& gm, int const lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        ::ExplicitDeposition(m_all_plasmas[i], fields, multi_laser, gm, lev);
    }
}

void
MultiPlasma::AdvanceParticles (
    const Fields & fields, const MultiLaser & multi_laser, amrex::Geometry const& gm,
    bool temp_slice, int lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        AdvancePlasmaParticles(m_all_plasmas[i], fields, gm, temp_slice,
                               lev, m_all_bins[i], multi_laser);
    }
}

void
MultiPlasma::ResetParticles (int lev)
{
    if (m_nplasmas < 1) return;
    for (auto& plasma : m_all_plasmas) {
        ResetPlasmaParticles(plasma, lev);
    }
}

void
MultiPlasma::DepositNeutralizingBackground (
    Fields & fields, const MultiLaser & multi_laser, int which_slice, amrex::Geometry const& gm, int const lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        if (m_all_plasmas[i].m_neutralize_background) {
            // current of ions is zero, so they are not deposited.
            ::DepositCurrent(m_all_plasmas[i], fields, multi_laser, which_slice, false,
                             false, true, false, gm, lev, m_all_bins[i], m_sort_bin_size);
        }
    }
}

void
MultiPlasma::DoFieldIonization (
    const int lev, const amrex::Geometry& geom, const Fields& fields)
{
    for (auto& plasma : m_all_plasmas) {
        plasma.IonizationModule(lev, geom, fields);
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
    constexpr int lev = 0;
    m_all_bins.clear();
    for (auto& plasma : m_all_plasmas) {
        m_all_bins.emplace_back(
            findParticlesInEachTile(lev, bx, m_sort_bin_size, plasma, geom));
    }
}

void
MultiPlasma::doCoulombCollision (int lev, amrex::Box bx, amrex::Geometry geom)
{
    HIPACE_PROFILE("MultiPlasma::doCoulombCollision");
    for (int i = 0; i < m_ncollisions; ++i)
    {
        AMREX_ALWAYS_ASSERT(lev == 0);
        auto& species1 = m_all_plasmas[ m_all_collisions[i].m_species1_index ];
        auto& species2 = m_all_plasmas[ m_all_collisions[i].m_species2_index ];

        // TODO: enable tiling

        CoulombCollision::doCoulombCollision(
            lev, bx, geom, species1, species2,
            m_all_collisions[i].m_isSameSpecies, m_all_collisions[i].m_CoulombLog);
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
MultiPlasma::TagByLevel (const int nlev, amrex::Vector<amrex::Geometry> geom3D, const int islice)
{
    for (auto& plasma : m_all_plasmas) {
        plasma.TagByLevel(nlev, geom3D, islice);
    }
}
