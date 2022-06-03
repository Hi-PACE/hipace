/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Remi Lehe
 * Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "BeamParticleContainer.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

namespace
{
    void QueryElementSetChargeMass (amrex::ParmParse& pp, amrex::Real& charge, amrex::Real& mass)
    {
        PhysConst phys_const = get_phys_const();

        std::string element = "electron";
        queryWithParser(pp, "element", element);
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
    queryWithParser(pp, "charge", m_charge);
    queryWithParser(pp, "mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The beam particle mass must not be 0");

    getWithParser(pp, "injection_type", m_injection_type);
    amrex::Vector<amrex::Real> tmp_vector;
    if (queryWithParser(pp, "ppc", tmp_vector)){
        AMREX_ALWAYS_ASSERT(tmp_vector.size() == AMREX_SPACEDIM);
        for (int i=0; i<AMREX_SPACEDIM; i++) m_ppc[i] = tmp_vector[i];
    }
    queryWithParser(pp, "dx_per_dzeta", m_dx_per_dzeta);
    queryWithParser(pp, "dy_per_dzeta", m_dy_per_dzeta);
    queryWithParser(pp, "duz_per_uz0_dzeta", m_duz_per_uz0_dzeta);
    queryWithParser(pp, "do_z_push", m_do_z_push);
    queryWithParser(pp, "insitu_sep", m_insitu_sep);
    queryWithParser(pp, "n_subcycles", m_n_subcycles);
    queryWithParser(pp, "finest_level", m_finest_level);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_n_subcycles >= 1, "n_subcycles must be >= 1");
    if (m_injection_type == "fixed_ppc" || m_injection_type == "from_file"){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( (m_dx_per_dzeta == 0.) && (m_dy_per_dzeta == 0.)
                                           && (m_duz_per_uz0_dzeta == 0.),
        "Tilted beams and correlated energy spreads are only implemented for fixed weight beams");
    }
}

amrex::Real
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    PhysConst phys_const = get_phys_const();
    amrex::Real ptime {0.};
    if (m_injection_type == "fixed_ppc") {

        amrex::ParmParse pp(m_name);
        getWithParser(pp, "zmin", m_zmin);
        getWithParser(pp, "zmax", m_zmax);
        getWithParser(pp, "radius", m_radius);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> position_mean{0., 0., 0.};
        queryWithParser(pp, "position_mean", position_mean);
        queryWithParser(pp, "min_density", m_min_density);
        amrex::Vector<int> random_ppc {false, false, false};
        queryWithParser(pp, "random_ppc", random_ppc);
        const GetInitialDensity get_density(m_name);
        const GetInitialMomentum get_momentum(m_name);
        InitBeamFixedPPC(m_ppc, get_density, get_momentum, geom, m_zmin,
                         m_zmax, m_radius, position_mean, m_min_density, random_ppc);

    } else if (m_injection_type == "fixed_weight") {

        amrex::ParmParse pp(m_name);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        bool can = false;
        amrex::Real zmin = 0, zmax = 0;
        std::string profile = "gaussian";
        queryWithParser(pp, "profile", profile);
        if (profile == "can") {
            can = true;
            getWithParser(pp, "zmin", zmin);
            getWithParser(pp, "zmax", zmax);
        } else if (profile == "gaussian") {
        } else {
            amrex::Abort("Only gaussian and can are supported with fixed_weight beam injection");
        }
        getWithParser(pp, "position_mean", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) m_position_mean[idim] = loc_array[idim];
        getWithParser(pp, "position_std", loc_array);
        for (int idim=0; idim<AMREX_SPACEDIM; ++idim) m_position_std[idim] = loc_array[idim];
        getWithParser(pp, "num_particles", m_num_particles);
        bool charge_is_specified = queryWithParser(pp, "total_charge", m_total_charge);
        bool peak_density_is_specified = queryWithParser(pp, "density", m_density);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( charge_is_specified + peak_density_is_specified == 1,
            "Please specify exlusively either total_charge or density of the beam");
        queryWithParser(pp, "do_symmetrize", m_do_symmetrize);
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
                            m_dy_per_dzeta, can, zmin, zmax);

    } else if (m_injection_type == "from_file") {
#ifdef HIPACE_USE_OPENPMD
        amrex::ParmParse pp(m_name);
        getWithParser(pp, "input_file", m_input_file);
        bool coordinates_specified = queryWithParser(pp, "file_coordinates_xyz", m_file_coordinates_xyz);
        bool n_0_specified = queryWithParser(pp, "plasma_density", m_plasma_density);
        queryWithParser(pp, "iteration", m_num_iteration);
        bool species_specified = queryWithParser(pp, "openPMD_species_name", m_species_name);
        if(!species_specified) {
            m_species_name = m_name;
        }

        if(!n_0_specified) {
            m_plasma_density = 0;
        }

        ptime = InitBeamFromFileHelper(m_input_file, coordinates_specified, m_file_coordinates_xyz,
                                       geom, m_plasma_density, m_num_iteration, m_species_name,
                                       species_specified);
#else
        amrex::Abort("beam particle injection via external_file requires openPMD support: "
                     "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif  // HIPACE_USE_OPENPMD
    } else {

        amrex::Abort("Unknown beam injection type. Must be fixed_ppc, fixed_weight or from_file\n");

    }

    {
        // Allocate memory for in-situ diagnostics
        int nslices = geom.Domain().length(2);
        m_insitu_rdata.resize(nslices*m_insitu_rnp, 0.);
        m_insitu_idata.resize(nslices*m_insitu_inp, 0.);
    }
    /* setting total number of particles, which is required for openPMD I/O */
    m_total_num_particles = TotalNumberOfParticles();

    return ptime;
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

void
BeamParticleContainer::InSituComputeDiags (int islice, const BeamBins& bins, int islice0)
{
    amrex::Print()<<"InSituComputeDiags\n";
    using namespace amrex::literals;
    amrex::AllPrint()<<"islice "<<islice<<'\n';
    BeamBins::index_type const * const indices = bins.permutationPtr();
    BeamBins::index_type const * const offsets = bins.offsetsPtr();
    BeamBins::index_type const cell_start = offsets[islice-islice0];
    BeamBins::index_type const cell_stop = offsets[islice-islice0+1];
    int const num_particles = cell_stop-cell_start;
    amrex::Gpu::DeviceVector<amrex::Real> device_rdata;
    amrex::Gpu::DeviceVector<int> device_idata;
    device_rdata.resize(m_insitu_rnp);
    device_idata.resize(m_insitu_inp);
    for (int i=0; i<m_insitu_rnp; i++) device_rdata[i] = 0._rt;
    for (int i=0; i<m_insitu_inp; i++) device_idata[i] = 0._rt;
    amrex::Real* AMREX_RESTRICT p_rdata = device_rdata.data();
    int* AMREX_RESTRICT p_idata = device_idata.data();
    amrex::ParallelFor(
        num_particles,
        [=] AMREX_GPU_DEVICE (long idx) {
            const int ip = indices[cell_start+idx];
            amrex::Gpu::Atomic::Add(&p_rdata[0], 1.);
            amrex::Gpu::Atomic::Add(&p_rdata[1], 2.);
            amrex::Gpu::Atomic::Add(&p_idata[0], 1);
        });
    for (int i=0; i<m_insitu_rnp; i++) m_insitu_rdata[m_insitu_rnp*islice+i] = p_rdata[i];
    for (int i=0; i<m_insitu_inp; i++) {
        m_insitu_idata[m_insitu_inp*islice+i] = p_idata[i];
        amrex::AllPrint()<<m_insitu_inp*islice+i<<" is "<<m_insitu_idata[m_insitu_inp*islice+i]<<'\n';
    }
}

void
BeamParticleContainer::InSituWriteToFile (int step)
{
    amrex::Print()<<"InSituWriteToFile\n";
    using namespace amrex::literals;
    // open file

    std::ofstream ofs{"reduced_" + m_name + "." + std::to_string(step) + ".txt",
        std::ofstream::out | std::ofstream::app};

    // write step
    ofs << step;

    ofs << m_insitu_sep;

    // set precision
    ofs << std::fixed << std::setprecision(14) << std::scientific;

    // write time
    // ofs << WarpX::GetInstance().gett_new(0);

    // loop over data size and write
    for (const auto& item : m_insitu_idata) ofs << m_insitu_sep << item;
    for (const auto& item : m_insitu_rdata) ofs << m_insitu_sep << item;

    // end loop over data size

    // end line
    ofs << std::endl;

    // close file
    ofs.close();

    for (int i=0; i<m_insitu_rdata.size(); i++) m_insitu_rdata[i] = 0._rt;
    for (int i=0; i<m_insitu_idata.size(); i++) m_insitu_idata[i] = 0._rt;
}
