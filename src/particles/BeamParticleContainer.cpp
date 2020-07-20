#include "BeamParticleContainer.H"

void
BeamParticleContainer::ReadParameters ()
{
    amrex::ParmParse pp("beam");
    pp.get("zmin", m_zmin);
    pp.get("zmax", m_zmax);
    pp.get("radius", m_radius);
    pp.get("density", m_density);
    amrex::Vector<amrex::Real> tmp_vector;
    if (pp.queryarr("ppc", tmp_vector)){
        AMREX_ALWAYS_ASSERT(tmp_vector.size() == AMREX_SPACEDIM);
        for (int i=0; i<AMREX_SPACEDIM; i++) m_ppc[i] = tmp_vector[i];
    }
    pp.query("uz_mean", m_uz_mean);
    pp.query("u_std", m_u_std);
}

void
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();
    InitCanBeam(m_ppc, m_u_std, m_uz_mean, m_density, geom, m_zmin, m_zmax, m_radius);
}
