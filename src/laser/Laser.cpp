/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#include "Laser.H"
#include "utils/Parser.H"
#include "Hipace.H"

#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>
#include "particles/particles_utils/ShapeFactors.H"

#ifdef HIPACE_USE_OPENPMD
#include <openPMD/openPMD.hpp>
#endif

Laser::Laser (std::string name)
{
    m_name = name;
}

void
Laser::ReadParameters (const amrex::Geometry& laser_geom_3D)
{
    amrex::ParmParse pp(m_name);
    amrex::ParmParse pp_lasers("lasers");

    queryWithParser(pp, "init_type", m_laser_init_type);
    if (m_laser_init_type == "from_file") {
        HIPACE_PROFILE("MultiLaser::GetEnvelopeFromFile()");

        queryWithParser(pp, "input_file", m_input_file_path);
        queryWithParser(pp, "openPMD_laser_name", m_file_envelope_name);
        queryWithParser(pp, "iteration", m_file_num_iteration);
        if (Hipace::HeadRank()) {
            GetEnvelopeFromFile(laser_geom_3D);
        }

        // m_init_lambda0 is only read by the HeadRank, so we need to communicate it
#ifdef AMREX_USE_MPI
        MPI_Bcast(&m_init_lambda0,
                  1,
                  amrex::ParallelDescriptor::Mpi_typemap<decltype(m_init_lambda0)>::type(),
                  Hipace::HeadRankID(),
                  amrex::ParallelDescriptor::Communicator());
#endif

        if (m_init_lambda0 != 0.) {
            // lambda0 is read from input file, but it can be overwritten explicitly here
            queryWithParser(pp, "lambda0", m_init_lambda0);
        } else {
            // lambda0 not defined in file
            getWithParserAlt(pp, "lambda0", m_init_lambda0, pp_lasers);
        }
        return;
    }
    else if (m_laser_init_type == "gaussian") {
        queryWithParser(pp, "a0", m_a0);
        queryWithParser(pp, "w0", m_w0);
        queryWithParser(pp, "CEP", m_CEP);
        queryWithParser(pp, "propagation_angle_yz", m_propagation_angle_yz);
        queryWithParser(pp, "STC_theta_xy", m_STC_theta_xy);
        int length_is_specified = queryWithParser(pp, "L0", m_L0);
        int duration_is_specified = queryWithParser(pp, "tau", m_tau);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( length_is_specified + duration_is_specified == 1,
        "Please specify exclusively either the pulse length L0 or the duration tau of Gaussian lasers");
        if (duration_is_specified) m_L0 = m_tau * get_phys_const().c;
        if (length_is_specified) m_tau = m_L0 / get_phys_const().c;
        queryWithParser(pp, "focal_distance", m_focal_distance);
        queryWithParser(pp, "position_mean",  m_position_mean);
        queryWithParser(pp, "zeta",  m_zeta);
        queryWithParser(pp, "beta",  m_beta);
        queryWithParser(pp, "phi2",  m_phi2);
        getWithParser(pp_lasers, "lambda0", m_init_lambda0);
        return;
    }
    else if (m_laser_init_type == "parser") {
        std::string profile_real_str = "";
        std::string profile_imag_str = "";
        getWithParser(pp, "laser_real(x,y,z)", profile_real_str);
        getWithParser(pp, "laser_imag(x,y,z)", profile_imag_str);
        m_profile_real = makeFunctionWithParser<3>( profile_real_str, m_parser_lr, {"x", "y", "z"});
        m_profile_imag = makeFunctionWithParser<3>( profile_imag_str, m_parser_li, {"x", "y", "z"});
        getWithParser(pp_lasers, "lambda0", m_init_lambda0);
        return;
    }
    else {
        amrex::Abort("Illegal init type specified for laser. "
                     "Must be one of: gaussian, from_file, parser");
    }
}

void
Laser::GetEnvelopeFromFile (amrex::Geometry laser_geom_3D) {
#ifdef HIPACE_USE_OPENPMD
    // Check what kind of Datatype is used in the Laser file
    auto series = openPMD::Series( m_input_file_path , openPMD::Access::READ_ONLY );

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        series.iterations.contains(m_file_num_iteration),
        "Could not find iteration " + std::to_string(m_file_num_iteration) +
        " in file " + m_input_file_path + "\n"
    );

    auto iteration = series.iterations[m_file_num_iteration];

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        iteration.meshes.contains(m_file_envelope_name),
        "Could not find mesh '" + m_file_envelope_name + "' in file "
        + m_input_file_path + "\n"
    );

    auto mesh = iteration.meshes[m_file_envelope_name];

    // Check that we are reading a normalized vector potential and not an electric field
    const std::array<double, 7> units_file = mesh.unitDimension();
    const std::array<double, 7> units_norm_potential{0., 0., 0., 0., 0., 0., 0.};
    const std::array<double, 7> units_electric_field{1., 1., -3., -1., 0., 0., 0.};
    const std::string help_msg = "Make sure to store the normalized vector potential, "
        "set the Attribute 'envelopeField' to 'normalized_vector_potential' and "
        "unitDimension to '" + amrex::ToString(units_norm_potential) + "'. "
        "If you are using LASY to generate the laser, pass 'save_as_vector_potential=True' "
        "to laser.write_to_file() or write_to_openpmd_file()";

    if (mesh.containsAttribute("envelopeField")) {
        const std::string field_type = mesh.getAttribute("envelopeField").get<std::string>();
        if (field_type == "electric_field") {
            amrex::Abort("Attribute 'envelopeField' in file '" + m_input_file_path +
                "' is set to 'electric_field' which is not compatible with HiPACE++. " +
                help_msg
            );
        } else if (field_type != "normalized_vector_potential") {
            amrex::AllPrint() << "WARNING: Attribute 'envelopeField' in file '"
                << m_input_file_path << "' is set to '" << field_type << "' which is not "
                " recognized. " << help_msg << '\n';
        }
    }

    if (units_file == units_electric_field) {
        amrex::Abort("unitDimension '" + amrex::ToString(units_file) + "' in file '"
            + m_input_file_path + "' is that of an electric field which is not compatible "
            "with HiPACE++. " + help_msg
        );
    } else if (units_file != units_norm_potential) {
        amrex::AllPrint() << "WARNING: unitDimension '" << amrex::ToString(units_file)
            << "' in file '" << m_input_file_path << "' is not recognized. "
            << help_msg << '\n';
    }

    if (mesh.containsAttribute("angularFrequency")) {
        m_init_lambda0 = 2.*MathConst::pi*PhysConstSI::c
            / mesh.getAttribute("angularFrequency").get<double>();
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mesh.contains(openPMD::RecordComponent::SCALAR),
        "Could not find component '" +
        std::string(openPMD::RecordComponent::SCALAR) +
        "' in file " + m_input_file_path + "\n"
    );

    auto comp = mesh[openPMD::RecordComponent::SCALAR];

    const PhysConst phc = get_phys_const();
    const amrex::Real clight = phc.c;

    const auto extent = comp.getExtent();
    std::vector<double> offset = mesh.gridGlobalOffset();
    std::vector<double> position = comp.position<double>();
    std::vector<double> spacing = mesh.gridSpacing<double>();

    const std::vector<std::string> axis_labels = mesh.axisLabels();

    AMREX_ALWAYS_ASSERT(extent.size() >= 3);

    m_strides = {extent[2], extent[2] * extent[1]};
    m_bigend = {
        static_cast<int>(extent[2] - 1),
        static_cast<int>(extent[1] - 1),
        static_cast<int>(extent[0] - 1)
    };
    m_unitSI = static_cast<amrex::Real>(comp.unitSI());

    if (axis_labels.size() >= 3 &&
        axis_labels[0] == "t" && axis_labels[1] == "y" && axis_labels[2] == "x") {
        m_file_geometry = "xyt";
    } else if (axis_labels.size() >= 3 &&
               axis_labels[0] == "z" && axis_labels[1] == "y" && axis_labels[2] == "x") {
        m_file_geometry = "xyz";
    } else if (axis_labels.size() >= 2 && axis_labels[0] == "t" && axis_labels[1] == "r") {
        m_file_geometry = "rt";
    } else {
        amrex::Abort("Incorrect axis labels in laser file, must be either tyx, zyx or tr");
    }

    const int ndim = m_file_geometry.size();

    for (int i=0; i<ndim; ++i) {
        const int rdim = ndim-1-i; // convert from C to F order
        if (m_file_geometry[i] == 't') {
            m_dx_inv[i] = amrex::Real(-1. / (clight * spacing[rdim]));
            m_pos_offset[i] = amrex::Real(laser_geom_3D.ProbHi(Direction::z)
                                          - laser_geom_3D.CellSize(Direction::z) / 2);
        } else {
            m_dx_inv[i] = amrex::Real(1. / spacing[rdim]);
            m_pos_offset[i] = amrex::Real(offset[rdim] + spacing[rdim] * position[rdim]);
        }
    }

    const uint64_t num_cells = extent[0] * extent[1] * extent[2];

    const openPMD::Datatype input_type = comp.getDatatype();
    if (input_type == openPMD::Datatype::CFLOAT) {
        m_cf_laser_data.reset(
            reinterpret_cast<std::complex<float>*>(
                amrex::The_Pinned_Arena()->alloc(num_cells*sizeof(std::complex<float>))),
            [](std::complex<float> *p){
                amrex::The_Pinned_Arena()->free(reinterpret_cast<void*>(p)); });

        comp.loadChunk(m_cf_laser_data, {0u}, {-1u});

        m_cf_ptr = reinterpret_cast<amrex::GpuComplex<float>*>(m_cf_laser_data.get());
    } else if (input_type == openPMD::Datatype::CDOUBLE) {
        m_cd_laser_data.reset(
            reinterpret_cast<std::complex<double>*>(
                amrex::The_Pinned_Arena()->alloc(num_cells*sizeof(std::complex<double>))),
            [](std::complex<double> *p){
                amrex::The_Pinned_Arena()->free(reinterpret_cast<double*>(p)); });

        comp.loadChunk(m_cd_laser_data, {0u}, {-1u});

        m_cd_ptr = reinterpret_cast<amrex::GpuComplex<double>*>(m_cd_laser_data.get());
    } else {
        amrex::Abort("Unknown Datatype used in Laser input file. Must use CDOUBLE or CFLOAT\n");
    }

    series.flush();
#else
    amrex::Abort("loading a laser envelope from an external file requires openPMD support: "
                 "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif // HIPACE_USE_OPENPMD
}
