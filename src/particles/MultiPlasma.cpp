#include "MultiPlasma.H"
#include "particles/deposition/PlasmaDepositCurrent.H"
#include "particles/pusher/PlasmaParticleAdvance.H"

MultiPlasma::MultiPlasma (amrex::AmrCore* amr_core)
{

    amrex::ParmParse pp("plasmas");
    pp.getarr("names", m_names);
    pp.query("adaptive_density", m_adaptive_density);
    if (m_names[0] == "no_plasma") return;
    m_nplasmas = m_names.size();
    for (int i = 0; i < m_nplasmas; ++i) {
        m_all_plasmas.emplace_back(PlasmaParticleContainer(amr_core, m_names[i]));
    }
}

void
MultiPlasma::InitData (int lev, amrex::BoxArray slice_ba,
                       amrex::DistributionMapping slice_dm, amrex::Geometry slice_gm,
                       amrex::Geometry gm)
{
    for (auto& plasma : m_all_plasmas) {
        plasma.SetParticleBoxArray(lev, slice_ba);
        plasma.SetParticleDistributionMap(lev, slice_dm);
        plasma.SetParticleGeometry(lev, slice_gm);
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
            plasma.InitIonizationModule(gm, plasma_product);
        }
    }
}

amrex::Real
MultiPlasma::maxDensity ()
{
    amrex::Real max_density = 0;
    for (auto& plasma : m_all_plasmas) {
        max_density = amrex::max(max_density, plasma.m_density);
    }
    return amrex::max(max_density, m_adaptive_density);
}

void
MultiPlasma::DepositCurrent (
    Fields & fields, int which_slice, bool temp_slice, bool deposit_jx_jy, bool deposit_jz,
    bool deposit_rho, bool deposit_j_squared, amrex::Geometry const& gm, int const lev)
{
    for (auto& plasma : m_all_plasmas) {
        ::DepositCurrent(plasma, fields, which_slice, temp_slice, deposit_jx_jy, deposit_jz,
                         deposit_rho, deposit_j_squared, gm, lev);
    }
}

void
MultiPlasma::AdvanceParticles (
    Fields & fields, amrex::Geometry const& gm, bool temp_slice, bool do_push,
    bool do_update, bool do_shift, int lev)
{
    for (auto& plasma : m_all_plasmas) {
        AdvancePlasmaParticles(plasma, fields, gm, temp_slice, do_push, do_update, do_shift, lev);
    }
}

void
MultiPlasma::ResetParticles (int lev, bool initial)
{
    for (auto& plasma : m_all_plasmas) {
        ResetPlasmaParticles(plasma, lev, initial);
    }
}

void
MultiPlasma::DepositNeutralizingBackground (
    Fields & fields, int which_slice, amrex::Geometry const& gm, int const lev)
{
    for (int i=0; i<m_nplasmas; i++) {
        auto& plasma = m_all_plasmas[i];
        int ispecies =
            (which_slice == WhichSlice::PlasmaRhoIons || which_slice == WhichSlice::Plasma) ? i : 0;
        int nspecies =
            (which_slice == WhichSlice::PlasmaRhoIons || which_slice == WhichSlice::Plasma) ? m_nplasmas : 1;        
        if (plasma.m_neutralize_background){
            // current of ions is zero, so they are not deposited.
            ::DepositCurrent(plasma, fields, which_slice, false,
                             false, false, true, false, gm, lev,
                             ispecies, nspecies);
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
MultiPlasma::AllSpeciesNeutralizeBackground () const
{
    bool all_species_neutralize = true;
    for (auto& plasma : m_all_plasmas) {
        if (!plasma.m_neutralize_background) all_species_neutralize = false;
    }
    return all_species_neutralize;
}
