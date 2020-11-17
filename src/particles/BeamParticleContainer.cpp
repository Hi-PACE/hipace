#include "BeamParticleContainer.H"

void
BeamParticleContainer::ReadParameters ()
{
    amrex::ParmParse pp("beam");
    pp.get("zmin", m_zmin);
    pp.get("zmax", m_zmax);
    pp.get("radius", m_radius);
    pp.get("injection_type", m_injection_type);
    amrex::Vector<amrex::Real> tmp_vector;
    if (pp.queryarr("ppc", tmp_vector)){
        AMREX_ALWAYS_ASSERT(tmp_vector.size() == AMREX_SPACEDIM);
        for (int i=0; i<AMREX_SPACEDIM; i++) m_ppc[i] = tmp_vector[i];
    }
}

void
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();

    if (m_injection_type == "fixed_ppc") {

        const GetInitialDensity get_density;
        const GetInitialMomentum get_momentum;
        InitBeamFixedPPC(m_ppc, get_density, get_momentum, geom, m_zmin, m_zmax, m_radius);

    } else if (m_injection_type == "fixed_weight") {

        amrex::ParmParse pp("beam");
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        pp.get("position_mean", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) m_position_mean[idim] = loc_array[idim];
        pp.get("position_std", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) m_position_std[idim] = loc_array[idim];
        pp.get("num_particles", m_num_particles);
        pp.get("total_charge", m_total_charge);
        const GetInitialMomentum get_momentum;
        InitBeamFixedWeight(m_num_particles, get_momentum, m_position_mean,
                            m_position_std, m_total_charge);

    } else {

        amrex::Abort("Unknown beam injection type. Must be fixed_ppc or fixed_weight");

    }
}
