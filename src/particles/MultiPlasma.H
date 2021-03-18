#ifndef MULTIPLASMA_H_
#define MULTIPLASMA_H_

#include "PlasmaParticleContainer.H"
#include "fields/Fields.H"

class MultiPlasma
{

public:

    MultiPlasma (amrex::AmrCore* amr_core);

    ~MultiPlasma () {}

    void InitData (int lev, amrex::BoxArray slice_ba, amrex::DistributionMapping slice_dm, amrex::Geometry slice_gm);

    void DepositCurrent (
        Fields & fields, int which_slice, bool temp_slice, bool deposit_jx_jy, bool deposit_jz,
        bool deposit_rho, amrex::Geometry const& gm, int const lev);

    amrex::Real maxDensity ();

    void AdvanceParticles (
        Fields & fields, amrex::Geometry const& gm, bool temp_slice, bool do_push,
        bool do_update, bool do_shift, int lev);

    void ResetParticles (int lev, bool initial=false);

    void DepositNeutralizingBackground (
        Fields & fields, int which_slice, amrex::Geometry const& gm, int const lev);

private:

    amrex::Vector<PlasmaParticleContainer> m_all_plasmas; /**< contains all plasma containers */
    amrex::Vector<std::string> m_names; /**< names of all plasma containers */
    int m_nplasmas; /**< number of plasma containers */

};

#endif // MULTIPLASMA_H_
