#include "PlasmaParticleContainer.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

PlasmaParticleContainer::PlasmaParticleContainer (amrex::AmrCore* amr_core)
    : amrex::ParticleContainer<0,0,PlasmaIdx::nattribs>(amr_core->GetParGDB())
{
    amrex::ParmParse pp("plasma");
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

    m_num_exchange = TotalNumberOfParticles();
}
