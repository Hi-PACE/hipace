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
                       amrex::DistributionMapping slice_dm, amrex::Geometry slice_gm)
{
    for (auto& plasma : m_all_plasmas) {
        plasma.SetParticleBoxArray(lev, slice_ba);
        plasma.SetParticleDistributionMap(lev, slice_dm);
        plasma.SetParticleGeometry(lev, slice_gm);
        plasma.InitData();
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
    for (auto& plasma : m_all_plasmas) {
        if (plasma.m_neutralize_background){
            ::DepositCurrent(plasma, fields, which_slice, false, false, false, true, false, gm, lev);
        }
    }
}
