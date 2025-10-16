/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "GetInitialDensity.H"
#include "utils/Parser.H"
#include "fields/Fields.H"

#ifdef HIPACE_USE_OPENPMD
#include <openPMD/openPMD.hpp>
#endif

GetInitialDensity::GetInitialDensity (const std::string& name, amrex::Parser& parser)
{
    amrex::ParmParse pp(name);
    std::string profile;
    getWithParser(pp, "profile", profile);

    if (profile == "gaussian") {
        m_profile = BeamProfileType::Gaussian;
        getWithParser(pp, "density", m_density);
        m_density = std::abs(m_density);
        queryWithParser(pp, "position_mean", m_position_mean);
        queryWithParser(pp, "position_std", m_position_std);
    } else if (profile == "flattop") {
        m_profile = BeamProfileType::Flattop;
        getWithParser(pp, "density", m_density);
        m_density = std::abs(m_density);
    } else if (profile == "parsed") {
        m_profile = BeamProfileType::Parsed;
        std::string density_func_str = "0.";
        getWithParser(pp, "density(x,y,z)", density_func_str);
        m_density_func = makeFunctionWithParser<3>(density_func_str, parser, {"x", "y", "z"});
    } else {
        amrex::Abort("Unknown beam profile!");
    }
}

void
PlasmaDensityAccessor::define_parser (const amrex::ParserExecutor<3>& exe) {
    m_density_func = exe;
    m_profile_type = 0;
}

void
PlasmaDensityAccessor::define_from_file (const std::string& path, std::shared_ptr<float>& f_data,
                                         std::shared_ptr<double>& d_data) {
#ifdef HIPACE_USE_OPENPMD

    HIPACE_PROFILE("PlasmaParticleContainer::ReadDensityFile()");

    auto series = openPMD::Series(path, openPMD::Access::READ_ONLY);
    auto iteration = series.iterations.begin()->second;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        iteration.meshes.contains("density"),
        "Could not find mesh 'density' in file " + path + "\n"
    );

    auto mesh = iteration.meshes["density"];

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mesh.contains(openPMD::RecordComponent::SCALAR),
        "Could not find component '" +
        std::string(openPMD::RecordComponent::SCALAR) +
        "' in file " + path + "\n"
    );

    auto comp = mesh[openPMD::RecordComponent::SCALAR];

    auto extent = comp.getExtent();
    auto strides = extent;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mesh.dataOrder() == openPMD::Mesh::DataOrder::C,
        "Must use DataOrder::C in file " + path + "\n"
    );

    for (int i=static_cast<int>(strides.size())-1; i>=0; --i) {
        if (i == static_cast<int>(strides.size())-1) {
            strides[i] = 1;
        } else {
            strides[i] = strides[i+1] * extent[i+1];
        }
    }

    const std::vector<std::string> axis_labels = mesh.axisLabels();
    std::map<std::string, int> axis_labels_map;

    for (int i=0; i<static_cast<int>(axis_labels.size()); ++i) {
        axis_labels_map[axis_labels[i]] = i;
    }

    std::vector<double> offset = mesh.gridGlobalOffset();
    std::vector<double> position = comp.position<double>();
    std::vector<double> spacing = mesh.gridSpacing<double>();

    amrex::IntVect idx_perm;

    bool use_mode = false;
    std::uint64_t mode_stride = 0;
    std::uint64_t mode_bigend = 0;

    if (mesh.geometry() == openPMD::Mesh::Geometry::cartesian) {
        m_profile_type = 1;

        idx_perm[0] = axis_labels_map.count("x") > 0 ? axis_labels_map["x"] : -1;
        idx_perm[1] = axis_labels_map.count("y") > 0 ? axis_labels_map["y"] : -1;
        idx_perm[2] = axis_labels_map.count("z") > 0 ? axis_labels_map["z"] : -1;

        axis_labels_map.erase("x");
        axis_labels_map.erase("y");
        axis_labels_map.erase("z");
    } else if (mesh.geometry() == openPMD::Mesh::Geometry::thetaMode ||
               mesh.geometry() == openPMD::Mesh::Geometry::cylindrical) {
        m_profile_type = 3;

        if (axis_labels_map.size() + 1 == extent.size()) {
            use_mode = true;
            mode_stride = strides[0];
            mode_bigend = extent[0] - 1;
            extent.erase(extent.begin());
            strides.erase(strides.begin());
        }

        idx_perm[0] = axis_labels_map.count("r") > 0 ? axis_labels_map["r"] : -1;
        idx_perm[1] = axis_labels_map.count("z") > 0 ? axis_labels_map["z"] : -1;
        idx_perm[2] = -1;

        axis_labels_map.erase("r");
        axis_labels_map.erase("z");
    } else {
        amrex::Abort("Unknown geometry in file " + path + "\n");
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        axis_labels_map.size() == 0,
        "Unknown Axis label, must be subset of xyz or rz in file " + path + "\n"
    );

    for (int i=0; i<3; ++i) {
        m_strides[i] = idx_perm[i] != -1 ? strides[idx_perm[i]] : 0;
        m_bigend[i] = idx_perm[i] != -1 ? extent[idx_perm[i]] - 1 : 0;
        m_pos_offset[i] = idx_perm[i] != -1 ? static_cast<amrex::Real>(
            offset[idx_perm[i]] + spacing[idx_perm[i]] * position[idx_perm[i]]) : 0;
        m_dx_inv[i] = idx_perm[i] != -1 ? static_cast<amrex::Real>(1. / spacing[idx_perm[i]]) : 0;
    }

    if (use_mode) {
        m_strides[2] = mode_stride;
        m_bigend[2] = mode_bigend;
    }

    m_unitSI = static_cast<amrex::Real>(comp.unitSI());

    uint64_t num_cells = 1;
    for (int i=0; i<3; ++i) {
        num_cells *= (m_bigend[i] + 1);
    }

    auto input_type = comp.getDatatype();

    if (input_type == openPMD::Datatype::FLOAT) {

        f_data.reset(
            reinterpret_cast<float*>(amrex::The_Managed_Arena()->alloc(num_cells*sizeof(float))),
            [](float *p){ amrex::The_Managed_Arena()->free(reinterpret_cast<void*>(p)); });

        comp.loadChunk(f_data, {0u}, {-1u});

        m_f_ptr = f_data.get();

    } else if (input_type == openPMD::Datatype::DOUBLE) {

        m_profile_type += 1;

        d_data.reset(
            reinterpret_cast<double*>(amrex::The_Managed_Arena()->alloc(num_cells*sizeof(double))),
            [](double *p){ amrex::The_Managed_Arena()->free(reinterpret_cast<void*>(p)); });

        comp.loadChunk(d_data, {0u}, {-1u});

        m_d_ptr = d_data.get();

    } else {
        amrex::Abort("Unknown data type in file " + path + "\n");
    }

    series.flush();

#else
    amrex::Abort("loading a plasma density from an external file requires openPMD support: "
                 "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif
}
