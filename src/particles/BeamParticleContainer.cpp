#include "BeamParticleContainer.H"
#include "utils/Constants.H"
#include "Hipace.H"

#ifdef AMREX_USE_MPI
namespace {
    constexpr int comm_z_tag = 3000;
}
#endif

void
BeamParticleContainer::ReadParameters ()
{
    amrex::ParmParse pp(m_name);
    pp.get("injection_type", m_injection_type);
    amrex::Vector<amrex::Real> tmp_vector;
    if (pp.queryarr("ppc", tmp_vector)){
        AMREX_ALWAYS_ASSERT(tmp_vector.size() == AMREX_SPACEDIM);
        for (int i=0; i<AMREX_SPACEDIM; i++) m_ppc[i] = tmp_vector[i];
    }
    pp.query("dx_per_dzeta", m_dx_per_dzeta);
    pp.query("dy_per_dzeta", m_dy_per_dzeta);
    pp.query("do_z_push", m_do_z_push);
    if (m_injection_type == "fixed_ppc"){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( (m_dx_per_dzeta == 0.) && (m_dy_per_dzeta == 0.),
            "Tilted beams are not yet implemented for fixed ppc beams");
    }
}

void
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    reserveData();
    resizeData();

    PhysConst phys_const = get_phys_const();

    if (m_injection_type == "fixed_ppc") {

        amrex::ParmParse pp(m_name);
        pp.get("zmin", m_zmin);
        pp.get("zmax", m_zmax);
        pp.get("radius", m_radius);
        const GetInitialDensity get_density(m_name);
        const GetInitialMomentum get_momentum(m_name);
        InitBeamFixedPPC(m_ppc, get_density, get_momentum, geom, m_zmin, m_zmax, m_radius);

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

    } else {

        amrex::Abort("Unknown beam injection type. Must be fixed_ppc or fixed_weight");

    }
}

void
BeamParticleContainer::PassNumParticlesUpstreamRanks (MPI_Comm a_comm_z)
{
    const int my_rank_z = amrex::ParallelDescriptor::MyProc();
    
    if (my_rank_z >= 1)
    {
        const int num_local_particles = TotalNumberOfParticles(1,1); // get local number of particles
        const int upstream_particles = m_num_particles_on_upstream_ranks + num_local_particles;

        MPI_Send(&upstream_particles, 1,
                 amrex::ParallelDescriptor::Mpi_typemap<int>::type(),
                 my_rank_z-1, comm_z_tag, a_comm_z);
    }
}

void
BeamParticleContainer::RecvNumParticlesUpstreamRanks (MPI_Comm a_comm_z)
{
    const int my_rank_z = amrex::ParallelDescriptor::MyProc();
    if (my_rank_z  < amrex::ParallelDescriptor::NProcs()-1)
    {
        MPI_Status status;
        MPI_Recv(&m_num_particles_on_upstream_ranks, 1,
                 amrex::ParallelDescriptor::Mpi_typemap<int>::type(),
                 my_rank_z+1, comm_z_tag, a_comm_z, &status);
    }
}
