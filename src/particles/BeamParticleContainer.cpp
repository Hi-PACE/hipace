#include "BeamParticleContainer.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

namespace
{
    void QueryElementSetChargeMass (amrex::ParmParse& pp, amrex::Real& charge, amrex::Real& mass)
    {
        // normalized_units is directly queried here so we can defined the appropriate PhysConst
        // locally. We cannot use Hipace::m_phys_const as it has not been initialized when the
        // PlasmaParticleContainer constructor is called.
        amrex::ParmParse pph("hipace");
        bool normalized_units = false;
        pph.query("normalized_units", normalized_units);
        PhysConst phys_const = normalized_units ? make_constants_normalized() : make_constants_SI();

        std::string element = "electron";
        pp.query("element", element);
        if (element == "electron"){
            charge = -phys_const.q_e;
            mass = phys_const.m_e;
        } else if (element == "proton"){
            charge = phys_const.q_e;
            mass = phys_const.m_p;
        } else if (element == "positron"){
            charge = phys_const.q_e;
            mass = phys_const.m_e;
        }
    }
}


void
BeamParticleContainer::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    QueryElementSetChargeMass(pp, m_charge, m_mass);
    // Overwrite element's charge and mass if user specifies them explicitly
    pp.query("charge", m_charge);
    pp.query("mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The beam particle mass must not be 0");

    pp.get("injection_type", m_injection_type);
    amrex::Vector<amrex::Real> tmp_vector;
    if (pp.queryarr("ppc", tmp_vector)){
        AMREX_ALWAYS_ASSERT(tmp_vector.size() == AMREX_SPACEDIM);
        for (int i=0; i<AMREX_SPACEDIM; i++) m_ppc[i] = tmp_vector[i];
    }
    pp.query("dx_per_dzeta", m_dx_per_dzeta);
    pp.query("dy_per_dzeta", m_dy_per_dzeta);
    pp.query("duz_per_uz0_dzeta", m_duz_per_uz0_dzeta);
    pp.query("do_z_push", m_do_z_push);
    pp.query("n_subcycles", m_n_subcycles);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_n_subcycles >= 1, "n_subcycles must be >= 1");
    if (m_injection_type == "fixed_ppc" || m_injection_type == "from_file"){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( (m_dx_per_dzeta == 0.) && (m_dy_per_dzeta == 0.)
                                           && (m_duz_per_uz0_dzeta == 0.),
        "Tilted beams and correlated energy spreads are only implemented for fixed weight beams");
    }
}

void
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    PhysConst phys_const = get_phys_const();

    if (m_injection_type == "fixed_ppc") {

        amrex::ParmParse pp(m_name);
        pp.get("zmin", m_zmin);
        pp.get("zmax", m_zmax);
        pp.get("radius", m_radius);
        pp.query("min_density", m_min_density);
        const GetInitialDensity get_density(m_name);
        const GetInitialMomentum get_momentum(m_name);
        InitBeamFixedPPC(m_ppc, get_density, get_momentum, geom, m_zmin,
                         m_zmax, m_radius, m_min_density);

    } else if (m_injection_type == "fixed_weight") {

        amrex::ParmParse pp(m_name);
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
        if (m_do_symmetrize) AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_num_particles%4 == 0,
            "To symmetrize the beam, please specify a beam particle number divisible by 4.");

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

        const GetInitialMomentum get_momentum(m_name);
        InitBeamFixedWeight(m_num_particles, get_momentum, m_position_mean,
                            m_position_std, m_total_charge, m_do_symmetrize, m_dx_per_dzeta,
                            m_dy_per_dzeta);

    } else if (m_injection_type == "from_file") {
#ifdef HIPACE_USE_OPENPMD
        amrex::ParmParse pp(m_name);
        pp.get("input_file", m_input_file);
        bool coordinates_specified = pp.query("file_coordinates_xyz", m_file_coordinates_xyz);
        bool n_0_specified = pp.query("plasma_density", m_plasma_density);
        pp.query("iteration", m_num_iteration);
        bool species_specified = pp.query("openPMD_species_name", m_species_name);
        if(!species_specified) {
            m_species_name = m_name;
        }

        if(!n_0_specified) {
            m_plasma_density = 0;
        }

        InitBeamFromFileHelper(m_input_file, coordinates_specified, m_file_coordinates_xyz, geom,
                          m_plasma_density, m_num_iteration, m_species_name, species_specified);
#else
        amrex::Abort("beam particle injection via external_file requires openPMD support: "
                     "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif  // HIPACE_USE_OPENPMD
    } else {

        amrex::Abort("Unknown beam injection type. Must be fixed_ppc, fixed_weight or from_file\n");

    }

    /* setting total number of particles, which is required for openPMD I/O */
    m_total_num_particles = TotalNumberOfParticles();

}

amrex::Long BeamParticleContainer::TotalNumberOfParticles (bool only_valid, bool only_local) const
{
    amrex::Long nparticles = 0;
    if (only_valid) {
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<unsigned long long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        auto const& ptaos = this->GetArrayOfStructs();
        ParticleType const* pp = ptaos().data();

        reduce_op.eval(ptaos.numParticles(), reduce_data,
                       [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
                       {
                           return (pp[i].id() > 0) ? 1 : 0;
                       });
        nparticles = static_cast<amrex::Long>(amrex::get<0>(reduce_data.value()));
    }
    else {
        nparticles = this->numParticles();
    }

    if (!only_local) {
        amrex::ParallelAllReduce::Sum(nparticles, amrex::ParallelContext::CommunicatorSub());
    }

    return nparticles;
}
