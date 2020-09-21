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
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();
    m_u_std = emittance / m_radius;
    GetInitialMomentum get_momentum(m_momentum_profile, m_u_mean, m_u_std);
    GetInitialDensity get_density(m_momentum_profile, m_radius, m_density);
    
    InitCanBeam(m_ppc, get_momentum, get_density, geom, m_zmin, m_zmax, m_radius);
}
