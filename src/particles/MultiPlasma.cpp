#include "MultiPlasma.H"
#include "particles/deposition/PlasmaDepositCurrent.H"
#include "particles/pusher/PlasmaParticleAdvance.H"
#include "TileSort.H"
#include "utils/HipaceProfilerWrapper.H"
#include "Hipace.H"

MultiPlasma::MultiPlasma (amrex::AmrCore* amr_core)
{

    amrex::ParmParse pp("plasmas");
    getWithParser(pp, "names", m_names);
    queryWithParser(pp, "adaptive_density", m_adaptive_density);
    queryWithParser(pp, "sort_bin_size", m_sort_bin_size);
    m_nominal_density = Hipace::m_normalized_units ? 1. : 1.e23;
    queryWithParser(pp, "nominal_density", m_nominal_density);
    queryWithParser(pp, "collisions", m_collision_names);

    if (m_names[0] == "no_plasma") return;
    m_nplasmas = m_names.size();
    for (int i = 0; i < m_nplasmas; ++i) {
        m_all_plasmas.emplace_back(PlasmaParticleContainer(amr_core, m_names[i]));
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
        const int lev = plasma.m_level;
        plasma.SetParticleBoxArray(lev, slice_ba[lev]);
        plasma.SetParticleDistributionMap(lev, slice_dm[lev]);
        plasma.SetParticleGeometry(lev, slice_gm[lev]);
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
            plasma.InitIonizationModule(gm[lev], plasma_product);
        }
    }
    if (m_nplasmas > 0) m_all_bins.resize(m_nplasmas);
}

amrex::Real
MultiPlasma::maxDensity () const
{
    amrex::Real max_density = 0;
    const amrex::Real c_t = get_phys_const().c * Hipace::m_physical_time;
    for (auto& plasma : m_all_plasmas) {
        max_density = amrex::max<amrex::Real>(max_density, plasma.m_density_func(0., 0., c_t));
    }
    return amrex::max(max_density, m_adaptive_density);
}

void
MultiPlasma::CheckDensity () const
{
    amrex::Real real_epsilon = std::numeric_limits<amrex::Real>::epsilon();
    if (maxDensity()/m_nominal_density < 1.e3 * real_epsilon ) {
        amrex::Print()<<"WARNING: The on-axis plasma density at z = " <<
            get_phys_const().c * Hipace::m_physical_time <<
            " is " << maxDensity() << ", which is much lower than the nominal density of " <<
            m_nominal_density <<". This is fine if this density is much below the highest density "
            "in the simulation. Otherwise, consider setting plasmas.nominal_density to the typical "
            "density used in your simulation.\n";
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        maxDensity() / m_nominal_density < 1.e-3/real_epsilon,
        "Density much higher than nominal density. Consider increasing plasmas.nominal_density");
    return;
}

void
MultiPlasma::DepositCurrent (
    Fields & fields, int which_slice, bool temp_slice, bool deposit_jx_jy, bool deposit_jz,
    bool deposit_rho, bool deposit_j_squared, amrex::Geometry const& gm, int const lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        ::DepositCurrent(m_all_plasmas[i], fields, which_slice, temp_slice,
                         deposit_jx_jy, deposit_jz, deposit_rho, deposit_j_squared,
                         gm, lev, m_all_bins[i], m_sort_bin_size);
    }
}

void
MultiPlasma::AdvanceParticles (
    Fields & fields, amrex::Geometry const& gm, bool temp_slice, bool do_push,
    bool do_update, bool do_shift, int lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        AdvancePlasmaParticles(m_all_plasmas[i], fields, gm, temp_slice,
                               do_push, do_update, do_shift, lev, m_all_bins[i]);
    }
}

void
MultiPlasma::ResetParticles (int lev, bool initial)
{
    if (m_nplasmas < 1) return;
    for (auto& plasma : m_all_plasmas) {
        ResetPlasmaParticles(plasma, lev, initial);
    }
}

void
MultiPlasma::DepositNeutralizingBackground (
    Fields & fields, int which_slice, amrex::Geometry const& gm, int const nlev)
{
    for (int lev = 0; lev < nlev; ++lev) {
        for (int i=0; i<m_nplasmas; i++) {
            if (m_all_plasmas[i].m_neutralize_background){
                // current of ions is zero, so they are not deposited.
                ::DepositCurrent(m_all_plasmas[i], fields, which_slice, false, false, false,
                                 true, false, gm, lev, m_all_bins[i], m_sort_bin_size);
            }
        }
    }
}

void
MultiPlasma::DoFieldIonization (
    const int lev, const amrex::Geometry& geom, Fields& fields)
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
MultiPlasma::AllSpeciesNeutralizeBackground () const
{
    bool all_species_neutralize = true;
    for (auto& plasma : m_all_plasmas) {
        if (!plasma.m_neutralize_background) all_species_neutralize = false;
    }
    return all_species_neutralize;
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

amrex::RealVect
MultiPlasma::GetUStd () const
{
    amrex::RealVect u_std = {0.,0.,0.};

    for (auto& plasma : m_all_plasmas) {
        u_std = plasma.GetUStd();
        if (m_nplasmas > 1) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE( (std::abs(u_std[0]) + std::abs(u_std[1])
                                               + std::abs(u_std[2])) < 1e-7,
                "Cannot use explicit solver + multiple plasma species + non-zero temperature");
        }
    }

    return u_std;
}

void
MultiPlasma::doCoulombCollision (int lev, amrex::Box bx, amrex::Geometry geom)
{
    HIPACE_PROFILE("MultiPlasma::doCoulombCollision");
    AMREX_ALWAYS_ASSERT(lev == 0);
    for (int i = 0; i < m_ncollisions; ++i)
    {
        auto& species1 = m_all_plasmas[ m_all_collisions[i].m_species1_index ];
        auto& species2 = m_all_plasmas[ m_all_collisions[i].m_species2_index ];

        // TODO: enable tiling

        CoulombCollision::doCoulombCollision(
            lev, bx, geom, species1, species2,
            m_all_collisions[i].m_isSameSpecies, m_all_collisions[i].m_CoulombLog);
    }
}
