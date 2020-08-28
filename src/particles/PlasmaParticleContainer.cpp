#include "PlasmaParticleContainer.H"

PlasmaParticleContainer::PlasmaParticleContainer (amrex::AmrCore* amr_core)
    : amrex::ParticleContainer<0,0,PlasmaIdx::nattribs>(amr_core->GetParGDB())
{
    amrex::ParmParse pp("plasma");
    pp.query("density", m_density);
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
PlasmaParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();

    const int dir = AMREX_SPACEDIM-1;
    const amrex::Real dx = geom.CellSize(dir);
    const amrex::Real hi = geom.ProbHi(dir);
    const amrex::Real lo = hi - dx;

    amrex::RealBox particleBox = geom.ProbDomain();
    particleBox.setHi(dir, hi);
    particleBox.setLo(dir, lo);

    InitParticles(m_ppc,m_u_std, m_u_mean, m_density, geom, particleBox);
}
