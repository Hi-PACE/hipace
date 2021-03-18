#include "Hipace.H"
#include "PlasmaParticleContainer.H"
#include "utils/HipaceProfilerWrapper.H"

namespace
{
    bool QueryElementSetChargeMass (amrex::ParmParse& pp, amrex::Real& charge, amrex::Real mass)
    {
        amrex::Abort("This function cannot be called before Hipace::m_phys_const is initialized");
        const PhysConst phys_const = get_phys_const();
        std::string element;
        bool element_is_specified = pp.query("element", element);
        if (element_is_specified){
            if (element == "electron"){
                charge = -phys_const.q_e;
                mass = phys_const.m_e;
            } else if (element == "H"){
                charge = phys_const.q_e;
                mass = phys_const.m_p;
            } else {
                amrex::Abort("unknown plasma species. Options are: electron and H.");
            }
    }
        return element_is_specified;
    }
}

//PlasmaParticleContainer::PlasmaParticleContainer (amrex::AmrCore* amr_core)
//    : amrex::ParticleContainer<0,0,PlasmaIdx::nattribs>(amr_core->GetParGDB())
void
PlasmaParticleContainer::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    pp.query("charge", m_charge); // TODO this should be pp.get
    pp.query("mass", m_mass); // TODO this should be pp.get
    // Below is the right way to specify charge+mass OR chemical element, but it currently does not
    // work because it uses Hipace::m_phys_const before it is initialized, resulting in random
    // segfault error.
    //
    // AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    //     QueryElementSetChargeMass(pp, m_charge, m_mass) ^
    //     (pp.query("charge", m_charge) && pp.query("mass", m_mass)),
    //     "Plasma: must specify EITHER <species>.element OR <species>.charge and <species>.mass");
    pp.query("neutralize_background", m_neutralize_background);
    pp.query("density", m_density);
    pp.query("radius", m_radius);
    pp.query("max_qsa_weighting_factor", m_max_qsa_weighting_factor);
    amrex::Vector<amrex::Real> tmp_vector;
    if (pp.queryarr("ppc", tmp_vector)){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(tmp_vector.size() == AMREX_SPACEDIM-1,
                                         "ppc is only specified in transverse directions for plasma particles, it is 1 in the longitudinal direction z. Hence, in 3D, plasma.ppc should only contain 2 values");
        for (int i=0; i<AMREX_SPACEDIM-1; i++) m_ppc[i] = tmp_vector[i];
    }
    amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
    if (pp.query("u_mean", loc_array)) {
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            m_u_mean[idim] = loc_array[idim];
        }
    }
    if (pp.query("u_std", loc_array)) {
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            m_u_std[idim] = loc_array[idim];
        }
    }
}

void
PlasmaParticleContainer::InitData ()
{
    reserveData();
    resizeData();

    InitParticles(m_ppc,m_u_std, m_u_mean, m_density, m_radius);

    m_num_exchange = TotalNumberOfParticles();
}
