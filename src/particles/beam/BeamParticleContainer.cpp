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
#include "utils/DeprecatedInput.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/InsituUtil.H"
#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/auxiliary/Filesystem.hpp>
#endif

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
    amrex::ParmParse pp_alt("beams");
    QueryElementSetChargeMass(pp, m_charge, m_mass);
    // Overwrite element's charge and mass if user specifies them explicitly
    queryWithParser(pp, "charge", m_charge);
    queryWithParser(pp, "mass", m_mass);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass != 0, "The beam particle mass must not be 0");

    DeprecatedInput(m_name, "dx_per_dzeta", "position_mean = \"x_center+(z-z_center)"
        "*dx_per_dzeta\" \"y_center+(z-z_center)*dy_per_dzeta\" \"z_center\"");
    DeprecatedInput(m_name, "dy_per_dzeta", "position_mean = \"x_center+(z-z_center)"
        "*dx_per_dzeta\" \"y_center+(z-z_center)*dy_per_dzeta\" \"z_center\"");

    getWithParser(pp, "injection_type", m_injection_type);
    queryWithParser(pp, "duz_per_uz0_dzeta", m_duz_per_uz0_dzeta);
    queryWithParser(pp, "do_z_push", m_do_z_push);
    queryWithParserAlt(pp, "do_radiation_reaction", m_do_radiation_reaction, pp_alt);
    queryWithParserAlt(pp, "insitu_period", m_insitu_period, pp_alt);
    queryWithParserAlt(pp, "insitu_file_prefix", m_insitu_file_prefix, pp_alt);
    queryWithParserAlt(pp, "insitu_radius", m_insitu_radius, pp_alt);
    queryWithParser(pp, "n_subcycles", m_n_subcycles);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_n_subcycles >= 1, "n_subcycles must be >= 1");
    queryWithParser(pp, "do_salame", m_do_salame);
    queryWithParserAlt(pp, "reorder_period", m_reorder_period, pp_alt);
    amrex::Array<int, 2> idx_array
        {Hipace::m_depos_order_xy % 2, Hipace::m_depos_order_xy % 2};
    queryWithParserAlt(pp, "reorder_idx_type", idx_array, pp_alt);
    m_reorder_idx_type = amrex::IntVect(idx_array[0], idx_array[1], 0);
    amrex::Array<std::string, 3> field_str = {"0", "0", "0"};
    m_use_external_fields = queryWithParserAlt(pp, "external_E(x,y,z,t)", field_str, pp_alt);
    m_external_fields[0] = makeFunctionWithParser<4>(field_str[0], m_external_fields_parser[0],
        {"x", "y", "z", "t"});
    m_external_fields[1] = makeFunctionWithParser<4>(field_str[1], m_external_fields_parser[1],
        {"x", "y", "z", "t"});
    m_external_fields[2] = makeFunctionWithParser<4>(field_str[2], m_external_fields_parser[2],
        {"x", "y", "z", "t"});
    field_str = {"0", "0", "0"};
    m_use_external_fields = queryWithParserAlt(pp, "external_B(x,y,z,t)", field_str, pp_alt)
        || m_use_external_fields;
    m_external_fields[3] = makeFunctionWithParser<4>(field_str[0], m_external_fields_parser[3],
        {"x", "y", "z", "t"});
    m_external_fields[4] = makeFunctionWithParser<4>(field_str[1], m_external_fields_parser[4],
        {"x", "y", "z", "t"});
    m_external_fields[5] = makeFunctionWithParser<4>(field_str[2], m_external_fields_parser[5],
        {"x", "y", "z", "t"});
    if (m_injection_type == "fixed_ppc" || m_injection_type == "from_file"){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_duz_per_uz0_dzeta == 0.,
        "Tilted beams and correlated energy spreads are only implemented for fixed weight beams");
    }
    queryWithParserAlt(pp, "initialize_on_cpu", m_initialize_on_cpu, pp_alt);
    auto& soa = getBeamInitSlice().GetStructOfArrays();
    soa.GetIdCPUData().setArena(
        m_initialize_on_cpu ? amrex::The_Pinned_Arena() : amrex::The_Arena());
    for (int rcomp = 0; rcomp < soa.NumRealComps(); ++rcomp) {
        soa.GetRealData()[rcomp].setArena(
            m_initialize_on_cpu ? amrex::The_Pinned_Arena() : amrex::The_Arena());
    }
    for (int icomp = 0; icomp < soa.NumIntComps(); ++icomp) {
        soa.GetIntData()[icomp].setArena(
            m_initialize_on_cpu ? amrex::The_Pinned_Arena() : amrex::The_Arena());
    }
    queryWithParserAlt(pp, "do_spin_tracking", m_do_spin_tracking, pp_alt);
    if (m_do_spin_tracking) {
        getWithParserAlt(pp, "initial_spin", m_initial_spin, pp_alt);
        queryWithParserAlt(pp, "spin_anom", m_spin_anom, pp_alt);
        for (auto& beam_tile : m_slices) {
            // Use 3 real and 0 int runtime components
            beam_tile.define(3, 0);
        }
    }
}

amrex::Real
BeamParticleContainer::InitData (const amrex::Geometry& geom)
{
    using namespace amrex::literals;
    amrex::ParmParse pp(m_name);
    amrex::ParmParse pp_alt("beams");
    amrex::Real ptime {0.};
    if (m_injection_type == "fixed_ppc") {

        queryWithParser(pp, "ppc", m_ppc);
        getWithParser(pp, "zmin", m_zmin);
        getWithParser(pp, "zmax", m_zmax);
        getWithParser(pp, "radius", m_radius);
        queryWithParser(pp, "position_mean", m_position_mean);
        queryWithParser(pp, "min_density", m_min_density);
        m_min_density = std::abs(m_min_density);
        queryWithParser(pp, "random_ppc", m_random_ppc);
        m_get_density = GetInitialDensity{m_name, m_density_parser};
        m_get_momentum =  GetInitialMomentum{m_name};
        InitBeamFixedPPC3D();

    } else if (m_injection_type == "fixed_weight") {

        std::string profile = "gaussian";
        queryWithParser(pp, "profile", profile);
        queryWithParser(pp, "radius", m_radius);
        if (profile == "can") {
            m_can_profile = true;
            getWithParser(pp, "zmin", m_zmin);
            getWithParser(pp, "zmax", m_zmax);
        } else if (profile == "gaussian") {
            queryWithParser(pp, "zmin", m_zmin);
            queryWithParser(pp, "zmax", m_zmax);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE( !m_do_salame ||
                (m_zmin != -std::numeric_limits<amrex::Real>::infinity() &&
                 m_zmax !=  std::numeric_limits<amrex::Real>::infinity()),
                "For the SALAME algorithm it is mandatory to either use a 'can' profile or "
                "'zmin' and 'zmax' with a gaussian profile");
        } else {
            amrex::Abort("Only gaussian and can are supported with fixed_weight beam injection");
        }

        std::array<std::string, 3> pos_mean_arr{"","",""};
        getWithParser(pp, "position_mean", pos_mean_arr);
        queryWithParser(pp, "z_foc", m_z_foc);
        m_pos_mean_x_func = makeFunctionWithParser<1>(pos_mean_arr[0], m_pos_mean_x_parser, {"z"});
        m_pos_mean_y_func = makeFunctionWithParser<1>(pos_mean_arr[1], m_pos_mean_y_parser, {"z"});
        Parser::fillWithParser(pos_mean_arr[2], m_pos_mean_z);

        getWithParser(pp, "position_std", m_position_std);
        getWithParser(pp, "num_particles", m_num_particles);
        bool charge_is_specified = queryWithParser(pp, "total_charge", m_total_charge);
        bool peak_density_is_specified = queryWithParser(pp, "density", m_density);
        if (charge_is_specified) AMREX_ALWAYS_ASSERT_WITH_MESSAGE( Hipace::m_normalized_units == 0,
            "The option 'beam.total_charge' is only valid in SI units."
            "Please either specify the peak density with '<beam name>.density', "
            "or set 'hipace.normalized_units = 0' to run in SI units, and update the input file accordingly.");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( charge_is_specified + peak_density_is_specified == 1,
            "Please specify exlusively either total_charge or density of the beam");
        queryWithParser(pp, "do_symmetrize", m_do_symmetrize);
        if (m_do_symmetrize) AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_num_particles%4 == 0,
            "To symmetrize the beam, please specify a beam particle number divisible by 4.");

        if (peak_density_is_specified)
        {
            m_total_charge = m_density*m_charge;
            for (int idim=0; idim<AMREX_SPACEDIM; ++idim)
            {
                m_total_charge *= m_position_std[idim] * sqrt(2. * MathConst::pi);
            }
        }
        if (Hipace::m_normalized_units)
        {
            m_total_charge *= geom.InvCellSize(0)*geom.InvCellSize(1)*geom.InvCellSize(2);
        }

        m_get_momentum = GetInitialMomentum{m_name};
        InitBeamFixedWeight3D();
        m_total_num_particles = m_num_particles;
        if (Hipace::HeadRank()) {
            m_init_sorter.sortParticlesByBox(m_z_array.dataPtr(), m_z_array.size(),
                                             m_initialize_on_cpu, geom);
        }

    } else if (m_injection_type == "fixed_weight_pdf") {

        std::string pdf_func_str = "";
        getWithParser(pp, "pdf", pdf_func_str);
        m_pdf_parsers.emplace_back();
        m_pdf_func = makeFunctionWithParser<1>(pdf_func_str, m_pdf_parsers.back(), {"z"});

        std::array<std::string, 2> pos_mean_arr{"",""};
        std::array<std::string, 2> pos_std_arr{"",""};
        getWithParser(pp, "position_mean", pos_mean_arr);
        getWithParser(pp, "position_std", pos_std_arr);
        for (int i=0; i<2; ++i) {
            m_pdf_parsers.emplace_back();
            m_pdf_pos_func[i] = makeFunctionWithParser<1>(pos_mean_arr[i], m_pdf_parsers.back(), {"z"});
            m_pdf_parsers.emplace_back();
            m_pdf_pos_func[i+2] = makeFunctionWithParser<1>(pos_std_arr[i], m_pdf_parsers.back(), {"z"});
        }

        std::array<std::string, 3> u_mean_arr{"","",""};
        std::array<std::string, 3> u_std_arr{"","",""};
        getWithParser(pp, "u_mean", u_mean_arr);
        getWithParser(pp, "u_std", u_std_arr);
        for (int i=0; i<3; ++i) {
            m_pdf_parsers.emplace_back();
            m_pdf_u_func[i] = makeFunctionWithParser<1>(u_mean_arr[i], m_pdf_parsers.back(), {"z"});
            m_pdf_parsers.emplace_back();
            m_pdf_u_func[i+3] = makeFunctionWithParser<1>(u_std_arr[i], m_pdf_parsers.back(), {"z"});
        }

        getWithParser(pp, "num_particles", m_num_particles);
        queryWithParser(pp, "radius", m_radius);
        queryWithParser(pp, "z_foc", m_z_foc);
        queryWithParser(pp, "do_symmetrize", m_do_symmetrize);
        queryWithParser(pp, "pdf_ref_ratio", m_pdf_ref_ratio);
        if (m_do_symmetrize) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_num_particles%4 == 0,
                "To symmetrize the beam, please specify a beam particle number divisible by 4.");
        }

        bool charge_is_specified = queryWithParser(pp, "total_charge", m_total_charge);
        m_peak_density_is_specified = queryWithParser(pp, "density", m_density);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( !charge_is_specified || !Hipace::m_normalized_units,
            "The option 'beam.total_charge' is only valid in SI units."
            "Please either specify the peak density with '<beam name>.density', "
            "or set 'hipace.normalized_units = 0' to run in SI units, and update the input file accordingly.");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( charge_is_specified + m_peak_density_is_specified == 1,
            "Please specify exlusively either total_charge or density of the beam");

        InitBeamFixedWeightPDF3D();
        m_total_num_particles = m_num_particles;

    } else if (m_injection_type == "from_file") {
#ifdef HIPACE_USE_OPENPMD
        getWithParserAlt(pp, "input_file", m_input_file, pp_alt);
        bool coordinates_specified = queryWithParserAlt(pp, "file_coordinates_xyz",
                                                        m_file_coordinates_xyz, pp_alt);
        queryWithParserAlt(pp, "plasma_density", m_plasma_density, pp_alt);
        queryWithParserAlt(pp, "iteration", m_num_iteration, pp_alt);
        bool species_specified = queryWithParser(pp, "openPMD_species_name", m_species_name);
        if(!species_specified) {
            m_species_name = m_name;
        }

        ptime = InitBeamFromFileHelper(m_input_file, coordinates_specified, m_file_coordinates_xyz,
                                       geom, m_plasma_density, m_num_iteration, m_species_name,
                                       species_specified);
        m_total_num_particles = getBeamInitSlice().size();
        if (Hipace::HeadRank()) {
            m_init_sorter.sortParticlesByBox(
                getBeamInitSlice().GetStructOfArrays().GetRealData(BeamIdx::z).dataPtr(),
                getBeamInitSlice().size(), m_initialize_on_cpu, geom);
        }
#else
        amrex::Abort("beam particle injection via external_file requires openPMD support: "
                     "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif  // HIPACE_USE_OPENPMD
    } else {

        amrex::Abort("Unknown beam injection type. Must be fixed_ppc, fixed_weight or from_file\n");

    }

#ifdef AMREX_USE_MPI
    MPI_Bcast(&m_total_num_particles,
              1,
              amrex::ParallelDescriptor::Mpi_typemap<decltype(m_total_num_particles)>::type(),
              Hipace::HeadRankID(),
              amrex::ParallelDescriptor::Communicator());
#endif

    if (m_insitu_period > 0) {
#ifdef HIPACE_USE_OPENPMD
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_insitu_file_prefix !=
            Hipace::GetInstance().m_openpmd_writer.m_file_prefix,
            "Must choose a different beam insitu file prefix compared to the full diagnostics");
#endif
        // Allocate memory for in-situ diagnostics
        m_nslices = geom.Domain().length(2);
        m_insitu_rdata.resize(m_nslices*m_insitu_nrp, 0.);
        m_insitu_idata.resize(m_nslices*m_insitu_nip, 0);
        m_insitu_sum_rdata.resize(m_insitu_nrp, 0.);
        m_insitu_sum_idata.resize(m_insitu_nip, 0);
        if (m_do_spin_tracking) {
            m_insitu_spin_data.resize(m_nslices*m_insitu_n_spin, 0.);
            m_insitu_sum_spin_data.resize(m_insitu_n_spin, 0.);
        }
    }

    return ptime;
}

void BeamParticleContainer::TagByLevel (const int current_N_level,
    amrex::Vector<amrex::Geometry> const& geom3D, const int which_slice)
{
    HIPACE_PROFILE("BeamParticleContainer::TagByLevel()");

    auto& soa = getBeamSlice(which_slice).GetStructOfArrays();
    const amrex::Real * const pos_x = soa.GetRealData(BeamIdx::x).data();
    const amrex::Real * const pos_y = soa.GetRealData(BeamIdx::y).data();
    int * const p_mr_level = soa.GetIntData(BeamIdx::mr_level).data();

    const int lev1_idx = std::min(1, current_N_level-1);
    const int lev2_idx = std::min(2, current_N_level-1);

    const amrex::Real lo_x_lev1 = geom3D[lev1_idx].ProbLo(0);
    const amrex::Real lo_x_lev2 = geom3D[lev2_idx].ProbLo(0);

    const amrex::Real hi_x_lev1 = geom3D[lev1_idx].ProbHi(0);
    const amrex::Real hi_x_lev2 = geom3D[lev2_idx].ProbHi(0);

    const amrex::Real lo_y_lev1 = geom3D[lev1_idx].ProbLo(1);
    const amrex::Real lo_y_lev2 = geom3D[lev2_idx].ProbLo(1);

    const amrex::Real hi_y_lev1 = geom3D[lev1_idx].ProbHi(1);
    const amrex::Real hi_y_lev2 = geom3D[lev2_idx].ProbHi(1);

    amrex::ParallelFor(getNumParticlesIncludingSlipped(which_slice),
        [=] AMREX_GPU_DEVICE (int ip) {
            const amrex::Real xp = pos_x[ip];
            const amrex::Real yp = pos_y[ip];

            if (current_N_level > 2 &&
                lo_x_lev2 < xp && xp < hi_x_lev2 &&
                lo_y_lev2 < yp && yp < hi_y_lev2) {
                // level 2
                p_mr_level[ip] = 2;
            } else if (current_N_level > 1 &&
                lo_x_lev1 < xp && xp < hi_x_lev1 &&
                lo_y_lev1 < yp && yp < hi_y_lev1) {
                // level 1
                p_mr_level[ip] = 1;
            } else {
                // level 0
                p_mr_level[ip] = 0;
            }
        }
    );
}

void
BeamParticleContainer::intializeSlice (int slice, int which_slice) {
    if (m_injection_type == "fixed_ppc") {
        InitBeamFixedPPCSlice(slice, which_slice);
    } else if (m_injection_type == "fixed_weight") {
        InitBeamFixedWeightSlice(slice, which_slice);
    } else if (m_injection_type == "fixed_weight_pdf") {
        InitBeamFixedWeightPDFSlice(slice, which_slice);
    } else {
        HIPACE_PROFILE("BeamParticleContainer::intializeSlice()");
        const int num_particles = m_init_sorter.m_box_counts_cpu[slice];

        resize(which_slice, num_particles, 0);

        auto ptd_init = getBeamInitSlice().getParticleTileData();
        auto ptd = getBeamSlice(which_slice).getParticleTileData();

        const int slice_offset = m_init_sorter.m_box_offsets_cpu[slice];
        const auto permutations = m_init_sorter.m_box_permutations.dataPtr();

        amrex::ParallelFor(num_particles,
            [=] AMREX_GPU_DEVICE (const int ip) {
                const int idx_src = permutations[slice_offset + ip];
                ptd.rdata(BeamIdx::x)[ip] = ptd_init.rdata(BeamIdx::x)[idx_src];
                ptd.rdata(BeamIdx::y)[ip] = ptd_init.rdata(BeamIdx::y)[idx_src];
                ptd.rdata(BeamIdx::z)[ip] = ptd_init.rdata(BeamIdx::z)[idx_src];
                ptd.rdata(BeamIdx::w)[ip] = ptd_init.rdata(BeamIdx::w)[idx_src];
                ptd.rdata(BeamIdx::ux)[ip] = ptd_init.rdata(BeamIdx::ux)[idx_src];
                ptd.rdata(BeamIdx::uy)[ip] = ptd_init.rdata(BeamIdx::uy)[idx_src];
                ptd.rdata(BeamIdx::uz)[ip] = ptd_init.rdata(BeamIdx::uz)[idx_src];

                ptd.idcpu(ip) = ptd_init.idcpu(idx_src);
                ptd.idata(BeamIdx::nsubcycles)[ip] = 0;
            }
        );
    }

    if (m_do_spin_tracking) {
        HIPACE_PROFILE("BeamParticleContainer::intializeSpin()");
        auto ptd = getBeamSlice(which_slice).getParticleTileData();

        const amrex::RealVect initial_spin_norm = m_initial_spin / m_initial_spin.vectorLength();

        amrex::ParallelFor(getNumParticles(which_slice),
            [=] AMREX_GPU_DEVICE (const int ip) {
                ptd.m_runtime_rdata[0][ip] = initial_spin_norm[0];
                ptd.m_runtime_rdata[1][ip] = initial_spin_norm[1];
                ptd.m_runtime_rdata[2][ip] = initial_spin_norm[2];
            }
        );
    }
}

void
BeamParticleContainer::resize (int which_slice, int num_particles, int num_slipped_particles) {
    HIPACE_PROFILE("BeamParticleContainer::resize()");

    m_num_particles_without_slipped[(which_slice + m_slice_permutation) % WhichBeamSlice::N] =
        num_particles;
    m_num_particles_with_slipped[(which_slice + m_slice_permutation) % WhichBeamSlice::N] =
        num_particles + num_slipped_particles;
    getBeamSlice(which_slice).resize(num_particles + num_slipped_particles);
}

void
BeamParticleContainer::ReorderParticles (int beam_slice, int step, amrex::Geometry& slice_geom) {
    if (m_reorder_period > 0 && step % m_reorder_period == 0) {
        HIPACE_PROFILE("BeamParticleContainer::ReorderParticles()");

        const int np = getNumParticles(beam_slice);
        const int np_total = getNumParticlesIncludingSlipped(beam_slice);
        auto& ptile = getBeamSlice(beam_slice);
        amrex::Gpu::DeviceVector<unsigned int> perm;
        amrex::PermutationForDeposition<unsigned int>(perm, np, ptile, slice_geom.Domain(),
                                                      slice_geom, m_reorder_idx_type);
        const unsigned int* permutations = perm.dataPtr();
        auto& soa = ptile.GetStructOfArrays();

        { // Create a scope for the temporary vector below
            BeamTile::RealVector tmp_real(np_total);
            for (int comp = 0; comp < soa.NumRealComps(); ++comp) {
                auto src = soa.GetRealData(comp).data();
                amrex::ParticleReal* dst = tmp_real.data();
                amrex::ParallelFor(np_total,
                    [=] AMREX_GPU_DEVICE (int i) {
                        dst[i] = i < np ? src[permutations[i]] : src[i];
                    });

                amrex::Gpu::streamSynchronize();
                soa.GetRealData(comp).swap(tmp_real);
            }
        }

        BeamTile::IntVector tmp_int(np_total);
        for (int comp = 0; comp < soa.NumIntComps(); ++comp) {
            auto src = soa.GetIntData(comp).data();
            int* dst = tmp_int.data();
            amrex::ParallelFor(np_total,
                [=] AMREX_GPU_DEVICE (int i) {
                    dst[i] = i < np ? src[permutations[i]] : src[i];
                });

            amrex::Gpu::streamSynchronize();
            soa.GetIntData(comp).swap(tmp_int);
        }
    }
}

void
BeamParticleContainer::InSituComputeDiags (int islice)
{
    HIPACE_PROFILE("BeamParticleContainer::InSituComputeDiags()");

    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT(m_insitu_rdata.size()>0 && m_insitu_idata.size()>0 &&
                        m_insitu_sum_rdata.size()>0 && m_insitu_sum_idata.size()>0);

    const amrex::Real insitu_radius_sq = m_insitu_radius * m_insitu_radius;
    const PhysConst phys_const = get_phys_const();
    const amrex::Real clight_inv = 1.0_rt/phys_const.c;
    const auto ptd = getBeamSlice(WhichBeamSlice::This).getParticleTileData();

    amrex::TypeMultiplier<amrex::ReduceOps, amrex::ReduceOpSum[m_insitu_nrp + m_insitu_nip]> reduce_op;
    amrex::TypeMultiplier<amrex::ReduceData, amrex::Real[m_insitu_nrp], int[m_insitu_nip]> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    reduce_op.eval(
        getNumParticles(WhichBeamSlice::This), reduce_data,
        [=] AMREX_GPU_DEVICE (int ip) -> ReduceTuple
        {
            const amrex::Real x = ptd.pos(0, ip);
            const amrex::Real y = ptd.pos(1, ip);
            const amrex::Real z = ptd.pos(2, ip);
            const amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip] * clight_inv; // proper velocity to u
            const amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip] * clight_inv;
            const amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip] * clight_inv;
            const amrex::Real w = ptd.rdata(BeamIdx::w)[ip];

            const amrex::Real uz_inv = uz == 0._rt ? 0._rt : 1._rt / uz;

            if (!ptd.id(ip).is_valid() || x*x + y*y > insitu_radius_sq) {
                return amrex::IdentityTuple(ReduceTuple{}, reduce_op);
            }
            const amrex::Real gamma = std::sqrt(1.0_rt + ux*ux + uy*uy + uz*uz);
            return {            // Tuple contains:
                w,              // 0    sum(w)
                w*x,            // 1    [x]
                w*x*x,          // 2    [x^2]
                w*y,            // 3    [y]
                w*y*y,          // 4    [y^2]
                w*z,            // 5    [z]
                w*z*z,          // 6    [z^2]
                w*ux,           // 7    [ux]
                w*ux*ux,        // 8    [ux^2]
                w*uy,           // 9    [uy]
                w*uy*uy,        // 10   [uy^2]
                w*uz,           // 11   [uz]
                w*uz*uz,        // 12   [uz^2]
                w*x*ux,         // 13   [x*ux]
                w*y*uy,         // 14   [y*uy]
                w*z*uz,         // 15   [z*uz]
                w*x*uy,         // 16   [x*uy]
                w*y*ux,         // 17   [y*ux]
                w*ux*uz_inv,    // 18   [ux/uz]
                w*uy*uz_inv,    // 19   [uy/uz]
                w*gamma,        // 20   [ga]
                w*gamma*gamma,  // 21   [ga^2]
                1               // 22   Np
            };
        });

    ReduceTuple a = reduce_data.value();
    const amrex::Real sum_w_inv = amrex::get<0>(a) <= 0._rt ? 0._rt : 1._rt / amrex::get<0>(a);

    amrex::constexpr_for<0, m_insitu_nrp>(
        [&] (auto idx) {
            m_insitu_rdata[islice + idx * m_nslices] = amrex::get<idx>(a) *
                // sum(w) is not multiplied by sum_w_inv
                ( idx == 0 ? 1 : sum_w_inv );
            m_insitu_sum_rdata[idx] += amrex::get<idx>(a);
        }
    );

    amrex::constexpr_for<0, m_insitu_nip>(
        [&] (auto idx) {
            m_insitu_idata[islice + idx * m_nslices] = amrex::get<m_insitu_nrp+idx>(a);
            m_insitu_sum_idata[idx] += amrex::get<m_insitu_nrp+idx>(a);
        }
    );

    if (m_do_spin_tracking) {
        amrex::TypeMultiplier<amrex::ReduceOps, amrex::ReduceOpSum[m_insitu_n_spin]> reduce_op_spin;
        amrex::TypeMultiplier<amrex::ReduceData, amrex::Real[m_insitu_n_spin]> reduce_data_spin(reduce_op_spin);
        using ReduceTupleSpin = typename decltype(reduce_data_spin)::Type;
        reduce_op_spin.eval(
            getNumParticles(WhichBeamSlice::This), reduce_data_spin,
            [=] AMREX_GPU_DEVICE (int ip) -> ReduceTupleSpin
            {
                const amrex::Real x = ptd.pos(0, ip);
                const amrex::Real y = ptd.pos(1, ip);
                const amrex::Real sx = ptd.m_runtime_rdata[0][ip];
                const amrex::Real sy = ptd.m_runtime_rdata[1][ip];
                const amrex::Real sz = ptd.m_runtime_rdata[2][ip];
                const amrex::Real w = ptd.rdata(BeamIdx::w)[ip];

                if (!ptd.id(ip).is_valid() || x*x + y*y > insitu_radius_sq) {
                    return amrex::IdentityTuple(ReduceTupleSpin{}, reduce_op_spin);
                }
                return {            // Tuple contains:
                    w*sx,           // 0    [sx]
                    w*sx*sx,        // 1    [sx^2]
                    w*sy,           // 2    [sy]
                    w*sy*sy,        // 3    [sy^2]
                    w*sz,           // 4    [sz]
                    w*sz*sz,        // 5    [sz^2]
                };
            });

        ReduceTupleSpin a_spin = reduce_data_spin.value();

        amrex::constexpr_for<0, m_insitu_n_spin>(
            [&] (auto idx) {
                m_insitu_spin_data[islice + idx * m_nslices] =
                    amrex::get<idx>(a_spin) * sum_w_inv;
                m_insitu_sum_spin_data[idx] += amrex::get<idx>(a_spin);
            }
        );
    }
}

void
BeamParticleContainer::InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom)
{
    HIPACE_PROFILE("BeamParticleContainer::InSituWriteToFile()");

#ifdef HIPACE_USE_OPENPMD
    // create subdirectory
    openPMD::auxiliary::create_directories(m_insitu_file_prefix);
#endif

    // zero pad the rank number;
    std::string::size_type n_zeros = 4;
    std::string rank_num = std::to_string(amrex::ParallelDescriptor::MyProc());
    std::string pad_rank_num = std::string(n_zeros-std::min(rank_num.size(), n_zeros),'0')+rank_num;

    // open file
    std::ofstream ofs{m_insitu_file_prefix + "/reduced_" + m_name + "." + pad_rank_num + ".txt",
        std::ofstream::out | std::ofstream::app | std::ofstream::binary};

    const amrex::Real sum_w0 = m_insitu_sum_rdata[0];
    const std::size_t nslices = static_cast<std::size_t>(m_nslices);
    const amrex::Real normalized_density_factor = Hipace::m_normalized_units ?
        geom.CellSizeArray().product() : 1; // dx * dy * dz in normalized units, 1 otherwise
    const int is_normalized_units = Hipace::m_normalized_units;

    // specify the structure of the data later available in python
    // avoid pointers to temporary objects as second argument, stack variables are ok
    amrex::Vector<insitu_utils::DataNode> all_data{
        {"time"    , &time},
        {"step"    , &step},
        {"n_slices", &m_nslices},
        {"charge"  , &m_charge},
        {"mass"    , &m_mass},
        {"z_lo"    , &geom.ProbLo()[2]},
        {"z_hi"    , &geom.ProbHi()[2]},
        {"normalized_density_factor", &normalized_density_factor},
        {"is_normalized_units", &is_normalized_units},
        {"[x]"     , &m_insitu_rdata[1*nslices], nslices},
        {"[x^2]"   , &m_insitu_rdata[2*nslices], nslices},
        {"[y]"     , &m_insitu_rdata[3*nslices], nslices},
        {"[y^2]"   , &m_insitu_rdata[4*nslices], nslices},
        {"[z]"     , &m_insitu_rdata[5*nslices], nslices},
        {"[z^2]"   , &m_insitu_rdata[6*nslices], nslices},
        {"[ux]"    , &m_insitu_rdata[7*nslices], nslices},
        {"[ux^2]"  , &m_insitu_rdata[8*nslices], nslices},
        {"[uy]"    , &m_insitu_rdata[9*nslices], nslices},
        {"[uy^2]"  , &m_insitu_rdata[10*nslices], nslices},
        {"[uz]"    , &m_insitu_rdata[11*nslices], nslices},
        {"[uz^2]"  , &m_insitu_rdata[12*nslices], nslices},
        {"[x*ux]"  , &m_insitu_rdata[13*nslices], nslices},
        {"[y*uy]"  , &m_insitu_rdata[14*nslices], nslices},
        {"[z*uz]"  , &m_insitu_rdata[15*nslices], nslices},
        {"[x*uy]"  , &m_insitu_rdata[16*nslices], nslices},
        {"[y*ux]"  , &m_insitu_rdata[17*nslices], nslices},
        {"[ux/uz]" , &m_insitu_rdata[18*nslices], nslices},
        {"[uy/uz]" , &m_insitu_rdata[19*nslices], nslices},
        {"[ga]"    , &m_insitu_rdata[20*nslices], nslices},
        {"[ga^2]"  , &m_insitu_rdata[21*nslices], nslices},
        {"sum(w)"  , &m_insitu_rdata[0], nslices},
        {"Np"      , &m_insitu_idata[0], nslices},
        {"average" , {
            {"[x]"   , &(m_insitu_sum_rdata[ 1] /= sum_w0)},
            {"[x^2]" , &(m_insitu_sum_rdata[ 2] /= sum_w0)},
            {"[y]"   , &(m_insitu_sum_rdata[ 3] /= sum_w0)},
            {"[y^2]" , &(m_insitu_sum_rdata[ 4] /= sum_w0)},
            {"[z]"   , &(m_insitu_sum_rdata[ 5] /= sum_w0)},
            {"[z^2]" , &(m_insitu_sum_rdata[ 6] /= sum_w0)},
            {"[ux]"  , &(m_insitu_sum_rdata[ 7] /= sum_w0)},
            {"[ux^2]", &(m_insitu_sum_rdata[ 8] /= sum_w0)},
            {"[uy]"  , &(m_insitu_sum_rdata[ 9] /= sum_w0)},
            {"[uy^2]", &(m_insitu_sum_rdata[10] /= sum_w0)},
            {"[uz]"  , &(m_insitu_sum_rdata[11] /= sum_w0)},
            {"[uz^2]", &(m_insitu_sum_rdata[12] /= sum_w0)},
            {"[x*ux]", &(m_insitu_sum_rdata[13] /= sum_w0)},
            {"[y*uy]", &(m_insitu_sum_rdata[14] /= sum_w0)},
            {"[z*uz]", &(m_insitu_sum_rdata[15] /= sum_w0)},
            {"[x*uy]", &(m_insitu_sum_rdata[16] /= sum_w0)},
            {"[y*ux]", &(m_insitu_sum_rdata[17] /= sum_w0)},
            {"[ux/uz]",&(m_insitu_sum_rdata[18] /= sum_w0)},
            {"[uy/uz]",&(m_insitu_sum_rdata[19] /= sum_w0)},
            {"[ga]"  , &(m_insitu_sum_rdata[20] /= sum_w0)},
            {"[ga^2]", &(m_insitu_sum_rdata[21] /= sum_w0)}
        }},
        {"total"   , {
            {"sum(w)", &m_insitu_sum_rdata[0]},
            {"Np"    , &m_insitu_sum_idata[0]}
        }}
    };

    if (m_do_spin_tracking) {
        const amrex::Vector<insitu_utils::DataNode> all_spin_data{
            {"[sx]"     , &m_insitu_spin_data[0*nslices], nslices},
            {"[sx^2]"   , &m_insitu_spin_data[1*nslices], nslices},
            {"[sy]"     , &m_insitu_spin_data[2*nslices], nslices},
            {"[sy^2]"   , &m_insitu_spin_data[3*nslices], nslices},
            {"[sz]"     , &m_insitu_spin_data[4*nslices], nslices},
            {"[sz^2]"   , &m_insitu_spin_data[5*nslices], nslices},
            {"average" , {
                {"[sx]"   , &(m_insitu_sum_spin_data[0] /= sum_w0)},
                {"[sx^2]" , &(m_insitu_sum_spin_data[1] /= sum_w0)},
                {"[sy]"   , &(m_insitu_sum_spin_data[2] /= sum_w0)},
                {"[sy^2]" , &(m_insitu_sum_spin_data[3] /= sum_w0)},
                {"[sz]"   , &(m_insitu_sum_spin_data[4] /= sum_w0)},
                {"[sz^2]" , &(m_insitu_sum_spin_data[5] /= sum_w0)}
            }}
        };

        insitu_utils::merge_data(all_data, all_spin_data);
    }

    if (ofs.tellp() == 0) {
        // write JSON header containing a NumPy structured datatype
        insitu_utils::write_header(all_data, ofs);
    }

    // write binary data according to datatype in header
    insitu_utils::write_data(all_data, ofs);

    // close file
    ofs.close();
    // assert no file errors
#ifdef HIPACE_USE_OPENPMD
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu beam diagnostics");
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ofs, "Error while writing insitu beam diagnostics. "
        "Maybe the specified subdirectory does not exist");
#endif

    // reset arrays for insitu data
    for (auto& x : m_insitu_rdata) x = 0.;
    for (auto& x : m_insitu_idata) x = 0;
    for (auto& x : m_insitu_sum_rdata) x = 0.;
    for (auto& x : m_insitu_sum_idata) x = 0;
    if (m_do_spin_tracking) {
        for (auto& x : m_insitu_spin_data) x = 0.;
        for (auto& x : m_insitu_sum_spin_data) x = 0.;
    }
}
