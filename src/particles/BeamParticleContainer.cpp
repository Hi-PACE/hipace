#include "BeamParticleContainer.H"
#include "Constants.H"
#include "Hipace.H"

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

    PhysConst phys_const = get_phys_const();

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
        bool charge_is_specified = pp.query("total_charge", m_total_charge);
        bool peak_density_is_specified = pp.query("density", m_density);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( charge_is_specified + peak_density_is_specified == 1,
        "Please specify exlusively either total_charge or density of the beam");
        pp.query("do_symmetrize", m_do_symmetrize);

        if (peak_density_is_specified)
        {
            m_total_charge = m_density*phys_const.q_e;
            for (int idim=0; idim<AMREX_SPACEDIM; ++idim)
            {
                m_total_charge *= m_position_std[idim] * sqrt(2. * MathConst::pi);
            }
        }
        if (Hipace::m_normalized_units)
        {
            auto dx = geom.CellSizeArray();
            m_total_charge /= dx[0]*dx[1]*dx[2];
        }

        const GetInitialMomentum get_momentum;
        InitBeamFixedWeight(m_num_particles, get_momentum, m_position_mean,
                            m_position_std, m_total_charge, m_do_symmetrize);

    } else {

        amrex::Abort("Unknown beam injection type. Must be fixed_ppc or fixed_weight");

    }
}
