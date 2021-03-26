#include "Hipace.H"
#include "PlasmaParticleContainer.H"
#include "utils/HipaceProfilerWrapper.H"

namespace
{
    bool QueryElementSetChargeMass (amrex::ParmParse& pp, amrex::Real& charge, amrex::Real& mass)
    {
        // normalized_units is directly queried here so we can defined the appropriate PhysConst
        // locally. We cannot use Hipace::m_phys_const as it has not been initialized when the
        // PlasmaParticleContainer constructor is called.
        amrex::ParmParse pph("hipace");
        bool normalized_units = false;
        pph.query("normalized_units", normalized_units);
        PhysConst phys_const = normalized_units ? make_constants_normalized() : make_constants_SI();

        std::string element;
        bool element_is_specified = pp.query("element", element);
        if (element_is_specified){
            if (element == "electron"){
                charge = -phys_const.q_e;
                mass = phys_const.m_e;
            } else if (element == "proton"){
                charge = phys_const.q_e;
                mass = phys_const.m_p;
            } else {
                amrex::Abort("unknown plasma species. Options are: electron and H.");
            }

        if( mass != phys_const.m_e ) {
            m_can_ionize = true;
        }
    }
        return element_is_specified;
    }
}

void
PlasmaParticleContainer::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    pp.query("charge", m_charge);
    pp.query("mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        QueryElementSetChargeMass(pp, m_charge, m_mass) ^
        (pp.query("charge", m_charge) && pp.query("mass", m_mass)),
        "Plasma: must specify EITHER <species>.element OR <species>.charge and <species>.mass");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The plasma particle mass must not be 0");

    pp.query("neutralize_background", m_neutralize_background);
    pp.query("density", m_density);
    pp.query("radius", m_radius);
    pp.query("channel_radius", m_channel_radius);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_channel_radius != 0,
                                     "The plasma channel radius must not be 0");
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

    InitParticles(m_ppc, m_u_std, m_u_mean, m_density, m_radius);

    if(m_can_ionize) {
        InitIonizationModule();
    }

    m_num_exchange = TotalNumberOfParticles();
}
