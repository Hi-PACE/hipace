/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "Diagnostic.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/DeprecatedInput.H"
#include "particles/deposition/HistogramDeposition.H"
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

void
Diagnostic::ReadParameters (int nlev, bool use_laser, bool has_beam)
{
    amrex::ParmParse ppd("diagnostic");
    amrex::ParmParse pph("hipace");

    // Make the default diagnostic objects, subset of: lev0, lev1, lev2, laser_diag
    amrex::Vector<std::string> diag_names{};
    for (int lev = 0; lev<nlev; ++lev) {
        std::string diag_name = "lev" + std::to_string(lev);
        diag_names.emplace_back(diag_name);
    }
    if (use_laser) {
        std::string diag_name = "laser_diag";
        diag_names.emplace_back(diag_name);
    }
    if (has_beam) {
        std::string diag_name = "beam_diag";
        diag_names.emplace_back(diag_name);
    }

    queryWithParser(ppd, "names", diag_names);
    if (diag_names.size() > 0 && diag_names[0] == "no_diag") {
        diag_names.clear();
    }

    m_diag_data.resize(diag_names.size());

    for(amrex::Long i = 0; i < m_diag_data.size(); ++i) {
        m_diag_data[i].m_diag_name = diag_names[i];
    }

#ifdef HIPACE_USE_OPENPMD
    m_openpmd_writer.ReadParameters();
#endif
}

bool
Diagnostic::needsRho () const {
    amrex::ParmParse ppd("diagnostic");
    for (auto& fd : m_diag_data) {
        amrex::ParmParse pp(fd.m_diag_name);
        amrex::Vector<std::string> comps{};
        queryWithParserAlt(pp, "field_data", comps, ppd);
        for (auto& c : comps) {
            if (c == "rho") {
                return true;
            }
        }
    }
    return false;
}

bool
Diagnostic::needsRhoIndividual () const {
    amrex::ParmParse ppd("diagnostic");
    for (auto& fd : m_diag_data) {
        amrex::ParmParse pp(fd.m_diag_name);
        amrex::Vector<std::string> comps{};
        queryWithParserAlt(pp, "field_data", comps, ppd);
        for (auto& c : comps) {
            // we don't know the names of all the plasmas here so just look for "rho_..."
            if (c.find("rho_") == 0) {
                return true;
            }
        }
    }
    return false;
}

bool
Diagnostic::needsTempIndividual () const {
    amrex::ParmParse ppd("diagnostic");
    for (auto& fd : m_diag_data) {
        amrex::ParmParse pp(fd.m_diag_name);
        amrex::Vector<std::string> comps{};
        queryWithParserAlt(pp, "field_data", comps, ppd);
        for (auto& c : comps) {
            // we don't know the names of all the plasmas here so just look for "ux_..."
            if (c.find("w_") == 0 ||
                c.find("ux_") == 0 || c.find("uy_") == 0 || c.find("uz_") == 0 ||
                c.find("ux^2_") == 0 || c.find("uy^2_") == 0 || c.find("uz^2_") == 0) {
                return true;
            }
        }
    }
    return false;
}

std::set<std::string>
Diagnostic::ParseCompsList (const amrex::Vector<std::string>& specified_comps,
                            const std::set<std::string>& available_comps,
                            std::map<std::string, bool>& is_global_comp_used,
                            bool use_local_comps, const std::string& error_msg)
{
    // set to store all used components to avoid duplicates
    std::set<std::string> comps_set{};

    // iterate through the user-provided components from left to right
    for (const std::string& comp_name : specified_comps) {
        if (comp_name == "all" || comp_name == "All") {
            is_global_comp_used[comp_name] = true;
            // insert all available components
            comps_set.insert(available_comps.begin(), available_comps.end());
        } else if (comp_name == "none" || comp_name == "None") {
            is_global_comp_used[comp_name] = true;
            // remove all components
            comps_set.clear();
        } else if (available_comps.count(comp_name) > 0) {
            is_global_comp_used[comp_name] = true;
            // insert requested component
            comps_set.insert(comp_name);
        } else if (comp_name.find("remove_") == 0 &&
                available_comps.count(
                comp_name.substr(std::string("remove_").size(), comp_name.size())) > 0) {
            is_global_comp_used[comp_name] = true;
            // remove requested component
            comps_set.erase(
                comp_name.substr(std::string("remove_").size(), comp_name.size()));
        } else if (use_local_comps) {
            // if field_data was specified through <diag name>,
            // assert that all components exist in the geometry
            amrex::Abort("Unknown diagnostics '" + comp_name + "' " + error_msg);
        } else {
            // if field_data was specified through diagnostic,
            // check later that all components are at least used by one of the diagnostics
            is_global_comp_used.try_emplace(comp_name, false);
        }
    }

    return comps_set;
}

void
Diagnostic::Initialize (int nlev, bool use_laser,
                        const amrex::Vector<std::string>& beam_names,
                        const amrex::Vector<std::string>& plasma_names)
{
    amrex::ParmParse ppd("diagnostic");
    amrex::ParmParse pph("hipace");

    // for each diagnostic object, choose a geometry and assign field_data

    // for the default diagnostics, what is the default geometry
    std::map<std::string, std::string> diag_name_to_default_geometry{};
    // for each geometry name, is it based on fields or laser
    std::map<std::string, DiagnosticData::diag_type> type_name_to_diag_type{};
    // for each geometry name, if its for fields what MR level is it on
    std::map<std::string, int> type_name_to_level{};
    // for each geometry, to which index do output components map to
    std::map<std::string, std::map<std::string, int>> type_name_to_output_comps_map{};
    // for each geometry, what output components are available
    std::map<std::string, std::set<std::string>> type_name_to_output_comps{};
    // in case there is an error, generate a string with all available geometries and components
    std::stringstream all_comps_error_str{};

    for (int lev = 0; lev<nlev; ++lev) {
        std::string diag_name = "lev" + std::to_string(lev);
        std::string type_name = "level_" + std::to_string(lev);
        diag_name_to_default_geometry.emplace(diag_name, type_name);
        type_name_to_diag_type.emplace(type_name, DiagnosticData::diag_type::field);
        type_name_to_level.emplace(type_name, lev);
        type_name_to_output_comps_map[type_name] = Comps[WhichSlice::This];
        // add derived diagnostics for Ex and Ey
        type_name_to_output_comps_map[type_name]["Ex"] = -1;
        type_name_to_output_comps_map[type_name]["Ey"] = -2;
    }
    if (use_laser) {
        std::string diag_name = "laser_diag";
        std::string type_name = "laser";
        diag_name_to_default_geometry.emplace(diag_name, type_name);
        type_name_to_diag_type.emplace(type_name, DiagnosticData::diag_type::laser);
        type_name_to_output_comps_map[type_name]["laserEnvelope"] = WhichLaserSlice::n00j00_r;
        // real=chi, imag=chi_initial
        type_name_to_output_comps_map[type_name]["laserChi"] = WhichLaserSlice::chi;
        // add derived diagnostics for |a^2|
        type_name_to_output_comps_map[type_name]["|a^2|"] = -1;
    }
    if (beam_names.size() > 0 || plasma_names.size() > 0) { // histogram
        std::string type_name = "histogram";
        type_name_to_diag_type.emplace(type_name, DiagnosticData::diag_type::histogram);
        for (amrex::Long i=0; i<beam_names.size(); ++i) {
            type_name_to_output_comps_map[type_name][beam_names[i]] = 0;
        }
        for (amrex::Long i=0; i<plasma_names.size(); ++i) {
            type_name_to_output_comps_map[type_name][plasma_names[i]] = 0;
        }
    }
    if (beam_names.size() > 0) {
        std::string diag_name = "beam_diag";
        std::string type_name = "beam";
        diag_name_to_default_geometry.emplace(diag_name, type_name);
        type_name_to_diag_type.emplace(type_name, DiagnosticData::diag_type::beam);
        for (amrex::Long i=0; i<beam_names.size(); ++i) {
            type_name_to_output_comps_map[type_name][beam_names[i]] = 0;
        }
    }
    if (plasma_names.size() > 0) {
        std::string type_name = "plasma_slice";
        type_name_to_diag_type.emplace(type_name, DiagnosticData::diag_type::plasma_slice);
        for (amrex::Long i=0; i<plasma_names.size(); ++i) {
            type_name_to_output_comps_map[type_name][plasma_names[i]] = 0;
        }
    }
    if (plasma_names.size() > 0) {
        std::string type_name = "particle_boundary";
        type_name_to_diag_type.emplace(type_name, DiagnosticData::diag_type::particle_boundary);
        for (amrex::Long i=0; i<beam_names.size(); ++i) {
            type_name_to_output_comps_map[type_name][beam_names[i]] = 0;
        }
        for (amrex::Long i=0; i<plasma_names.size(); ++i) {
            type_name_to_output_comps_map[type_name][plasma_names[i]] = 0;
        }
    }

    for (const auto& [type_name, comp_map] : type_name_to_output_comps_map) {
        all_comps_error_str << "Available components for type '"
            << type_name << "':\n    ";
        for (const auto& [comp, idx] : comp_map) {
            type_name_to_output_comps[type_name].insert(comp);
            all_comps_error_str << comp << " ";
        }
        all_comps_error_str << "\n";
    }
    all_comps_error_str << "Additionally, 'all' and 'none' are supported as field_data\n"
                        << "Components can be removed after 'all' by using 'remove_<comp name>'.\n";

    // keep track of all components from the input and later assert that they were all used
    std::map<std::string, bool> is_global_comp_used{};

    for (auto& fd : m_diag_data) {
        amrex::ParmParse pp(fd.m_diag_name);

        std::string base_type_name = "level_0";

        if (diag_name_to_default_geometry.count(fd.m_diag_name) > 0) {
            base_type_name = diag_name_to_default_geometry.at(fd.m_diag_name);
        }

        DeprecatedInput(fd.m_diag_name, "level", "type");
        if (queryWithParserAlt(pp, "base_geometry", base_type_name, ppd)) {
            amrex::Print() << "WARNING: '<diag name> or diagnostic.base_geometry' is deprecated! "
                "Use '<diag name> or diagnostic.type' instead!\n";
        }
        queryWithParser(pp, "type", base_type_name);

        if (type_name_to_diag_type.count(base_type_name) > 0) {
            fd.m_base_diag_type = type_name_to_diag_type.at(base_type_name);
        } else {
            amrex::Abort("Unknown diagnostics type: '" + base_type_name + "'!\n" +
                         all_comps_error_str.str());
        }

        if (fd.m_base_diag_type == DiagnosticData::diag_type::field) {
            fd.m_level = type_name_to_level.at(base_type_name);
        }

        // general parametes for all base geometries

        // hipace.output_period
        if (queryWithParser(pph, "output_period", fd.m_output_period.m_func_str)) {
            amrex::Print() << "WARNING: 'hipace.output_period' is deprecated! "
                "Use 'diagnostic.output_period' instead!\n";
        }
        // diagnostic.output_period
        queryWithParser(ppd, "output_period", fd.m_output_period.m_func_str);
        if (fd.m_base_diag_type == DiagnosticData::diag_type::beam) {
            // diagnostic.beam_output_period
            queryWithParser(ppd, "beam_output_period", fd.m_output_period.m_func_str);
        }
        // <diag_name>.output_period
        queryWithParser(pp, "output_period", fd.m_output_period.m_func_str);
        fd.m_output_period.compile();

        // parameters for all particle based diagnostics

        if (fd.m_base_diag_type == DiagnosticData::diag_type::beam ||
            fd.m_base_diag_type == DiagnosticData::diag_type::plasma_slice ||
            fd.m_base_diag_type == DiagnosticData::diag_type::particle_boundary)
        {
            queryWithParser(pp, "species", fd.m_species_names);

            if (fd.m_species_names.empty()) {
                // by default output all components
                fd.m_species_names.push_back("all");
            }

            std::set<std::string> comps_set = ParseCompsList(fd.m_species_names,
                type_name_to_output_comps[base_type_name],
                is_global_comp_used, true,
                "for species in type '" + base_type_name + "'!\n" +
                all_comps_error_str.str()
            );

            fd.m_species_names.assign(comps_set.begin(), comps_set.end());
            fd.m_comps_output = fd.m_species_names;
            fd.m_nfields = fd.m_species_names.size();
        }

        // beam parameters

        if (fd.m_base_diag_type == DiagnosticData::diag_type::beam) {
            queryWithParser(pp, "output_ratio", fd.m_output_ratio);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fd.m_output_ratio >= 1, "output_ratio must be >= 1");
        }

        // plasma slice parameters

        if (fd.m_base_diag_type == DiagnosticData::diag_type::plasma_slice) {
            queryWithParser(pp, "plasma_output_slice", fd.m_plasma_output_slice);
        }

        // parameters for all mesh based diagnostics

        if (fd.m_base_diag_type == DiagnosticData::diag_type::field ||
            fd.m_base_diag_type == DiagnosticData::diag_type::laser ||
            fd.m_base_diag_type == DiagnosticData::diag_type::histogram)
        {
            fd.m_use_custom_size_lo = queryWithParserAlt(pp, "patch_lo", fd.m_diag_lo, ppd);
            fd.m_use_custom_size_hi = queryWithParserAlt(pp, "patch_hi", fd.m_diag_hi, ppd);

            amrex::Array<int,3> diag_coarsen_arr{1,1,1};
            queryWithParserAlt(pp, "coarsening", diag_coarsen_arr, ppd);
            fd.m_diag_coarsen = amrex::IntVect(diag_coarsen_arr);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fd.m_diag_coarsen.min() >= 1,
                        "Coarsening ratio must be >= 1");

            queryWithParserAlt(pp, "include_ghost_cells", fd.m_include_ghost_cells, ppd);
        }

        // get histogram specific inputs

        if (fd.m_base_diag_type == DiagnosticData::diag_type::histogram)
        {
            getWithParser(pp, "hist_species_names", fd.m_hist_species_names);

            std::set<std::string> comps_set = ParseCompsList(fd.m_hist_species_names,
                type_name_to_output_comps[base_type_name],
                is_global_comp_used, true,
                "for hist_species_names in type '" + base_type_name + "'!\n" +
                all_comps_error_str.str()
            );
            fd.m_hist_species_names.assign(comps_set.begin(), comps_set.end());

            getWithParser(pp, "hist_num_bins", fd.m_hist_num_bins);
            getWithParser(pp, "hist_bins_lo", fd.m_hist_bins_lo);
            getWithParser(pp, "hist_bins_hi", fd.m_hist_bins_hi);
            bool add_z_axis = false;
            queryWithParser(pp, "hist_add_z_axis", add_z_axis);
            fd.m_integrate_along_z = !add_z_axis;
            queryWithParser(pp, "hist_exit_boundary", fd.m_hist_exit_boundary);

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                fd.m_hist_num_bins.size() == 1 || fd.m_hist_num_bins.size() == 2,
                "hist_num_bins must have either one or two values"
            );
            fd.m_hist_num_dims = fd.m_hist_num_bins.size();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                fd.m_hist_bins_lo.size() == fd.m_hist_num_dims &&
                fd.m_hist_bins_hi.size() == fd.m_hist_num_dims,
                "hist_bins_lo and hist_bins_hi must have the same "
                "number of values as hist_num_bins"
            );

            std::string func1;
            getWithParser(pp, "hist_function", func1);
            fd.m_hist_exe_q1 = makeFunctionWithParser<9>(func1, fd.m_hist_parser_q1,
                {"x", "y", "z", "ux", "uy", "uz", "ga_psi", "w", "ion_lev"});

            if (fd.m_hist_num_dims == 2) {
                std::string func2;
                getWithParser(pp, "hist_function2", func2);
                fd.m_hist_exe_q2 = makeFunctionWithParser<9>(func2, fd.m_hist_parser_q2,
                    {"x", "y", "z", "ux", "uy", "uz", "ga_psi", "w", "ion_lev"});

                if (fd.m_integrate_along_z) {
                    fd.m_remove_axis = {0, 0, 1};
                    fd.m_axis_labels = {func1, func2};
                } else {
                    fd.m_remove_axis = {0, 0, 0};
                    fd.m_axis_labels = {func1, func2, "z"};
                }
            } else {
                if (fd.m_integrate_along_z) {
                    fd.m_remove_axis = {0, 1, 1};
                    fd.m_axis_labels = {func1};
                } else {
                    fd.m_remove_axis = {0, 1, 0};
                    fd.m_axis_labels = {func1, "z"};
                }
            }

            std::string funcw = "w";
            queryWithParser(pp, "hist_weight", funcw);
            fd.m_hist_exe_w = makeFunctionWithParser<9>(funcw, fd.m_hist_parser_w,
                {"x", "y", "z", "ux", "uy", "uz", "ga_psi", "w", "ion_lev"});

            fd.m_nfields = fd.m_hist_species_names.size();
            fd.m_comps_output = fd.m_hist_species_names;

            fd.m_diag_coarsen[0] = 1;
            fd.m_diag_coarsen[1] = 1;
        }

        // get and parse dimensions and field_data parameter for field and laser diagnostics

        if (fd.m_base_diag_type == DiagnosticData::diag_type::field ||
            fd.m_base_diag_type == DiagnosticData::diag_type::laser)
        {
            std::string str_type;
            if (queryWithParserAlt(pp, "diag_type", str_type, ppd)) {
                amrex::Print() << "WARNING: '<diag name> or diagnostic.diag_type' is deprecated! "
                    "Use '<diag name> or diagnostic.dimensions' instead!\n";

            } else {
                getWithParserAlt(pp, "dimensions", str_type, ppd);
            }
            if (str_type == "xyz"){
                fd.m_remove_axis = {0, 0, 0};
                fd.m_axis_labels = {"x", "y", "z"};
            } else if (str_type == "xz") {
                fd.m_remove_axis = {0, 1, 0};
                fd.m_axis_labels = {"x", "z"};
            } else if (str_type == "yz") {
                fd.m_remove_axis = {1, 0, 0};
                fd.m_axis_labels = {"y", "z"};
            } else if (str_type == "xy_integrated") {
                fd.m_remove_axis = {0, 0, 1};
                fd.m_axis_labels = {"x", "y"};
                fd.m_integrate_along_z = true;
            } else {
                amrex::Abort("Unknown diagnostics type: must be xyz, xz, yz or xy_integrated.");
            }

            for (int i=0; i<3; ++i) {
                if (fd.m_remove_axis[i]) {
                    fd.m_diag_coarsen[i] = 1;
                }
            }

            amrex::Vector<std::string> use_comps{};
            const bool use_local_comps = queryWithParser(pp, "field_data", use_comps);
            if (!use_local_comps) {
                queryWithParser(ppd, "field_data", use_comps);
            }

            if (use_comps.empty()) {
                // by default output all components
                use_comps.push_back("all");
            }

            std::set<std::string> comps_set = ParseCompsList(use_comps,
                type_name_to_output_comps[base_type_name],
                is_global_comp_used, use_local_comps,
                "for field_data in type '" + base_type_name + "'!\n" +
                all_comps_error_str.str()
            );

            fd.m_comps_output.assign(comps_set.begin(), comps_set.end());
            fd.m_nfields = fd.m_comps_output.size();

            // copy the indexes of m_comps_output to the GPU
            fd.m_comps_output_idx.resize(fd.m_nfields);
            for (int i = 0; i < fd.m_nfields; ++i) {
                fd.m_comps_output_idx[i] =
                    type_name_to_output_comps_map.at(base_type_name).at(fd.m_comps_output[i]);
            }
            fd.m_comps_output_idx.copyToDeviceAsync();
        }
    }

    // check that all components are at least used by one of the diagnostics
    for (auto& [key, val] : is_global_comp_used) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(val,
            "Unknown or unused component in diagnostic.field_data.\n'" +
            key + "' does not belong to any diagnostic.names!\n" +
            all_comps_error_str.str()
        );
    }

    // if there are multiple diagnostic objects with the same m_comps_output (colliding component
    // names), append the name of the diagnostic object to the component name in the output
    std::map<std::string, int> num_occurrences;

    for (auto& fd : m_diag_data) {
        for (auto& comp_name : fd.m_comps_output) {
            num_occurrences[comp_name] += 1;
        }
    }

    for (auto& fd : m_diag_data) {
        bool has_collisions = false;
        for (auto& comp_name : fd.m_comps_output) {
            if (num_occurrences[comp_name] > 1) {
                has_collisions = true;
                break;
            }
        }
        if (has_collisions) {
            for (auto& comp_name : fd.m_comps_output) {
                comp_name += "_" + fd.m_diag_name;
            }
        }
    }
}

void
Diagnostic::InitDiagnosticsStep (amrex::Vector<amrex::Geometry>& field_geom,
                                 amrex::Geometry const& laser_geom,
                                 MultiPlasma& plasmas, MultiBeam& beams, int output_step,
                                 amrex::Real output_time, bool is_last_step)
{
#ifdef HIPACE_USE_OPENPMD
    if (hasAnyOutput(output_step, output_time, is_last_step)) {
        m_openpmd_writer.InitDiagnostics();
    }
#endif

    // all diagnostics that output a mesh

    for (auto& fd : m_diag_data) {

        if (!(fd.m_base_diag_type == DiagnosticData::diag_type::field ||
            fd.m_base_diag_type == DiagnosticData::diag_type::laser ||
            fd.m_base_diag_type == DiagnosticData::diag_type::histogram))
        {
            continue;
        }

        fd.m_has_output = hasOutput(fd, output_step, output_time, is_last_step);

        amrex::Geometry geom;

        // choose the geometry of the diagnostic
        switch (fd.m_base_diag_type) {
            case DiagnosticData::diag_type::field:
                geom = field_geom[fd.m_level];
                break;
            case DiagnosticData::diag_type::laser:
                geom = laser_geom;
                break;
            case DiagnosticData::diag_type::histogram:
                // particles are based on field level 0 geom
                geom = field_geom[0];
                break;
            default:
                break;
        }

        amrex::Box domain = geom.Domain();

        if (fd.m_include_ghost_cells) {
            switch (fd.m_base_diag_type) {
                case DiagnosticData::diag_type::field:
                    domain.grow(Hipace::GetInstance().m_fields.getSlices(fd.m_level).nGrowVect());
                    break;
                case DiagnosticData::diag_type::laser:
                    domain.grow(Hipace::GetInstance().m_multi_laser.getSlices().nGrowVect());
                    break;
                case DiagnosticData::diag_type::histogram:
                    domain.grow(Hipace::GetInstance().m_fields.getSlices(0).nGrowVect());
                    break;
                default:
                    break;
            }
        }

        const amrex::Box sim_domain = domain;
        amrex::Box cut_domain = domain;
        {
            // shrink box to user specified bounds m_diag_lo and m_diag_hi (in real space)
            const amrex::Real poff_x = GetPosOffset(0, geom, geom.Domain());
            const amrex::Real poff_y = GetPosOffset(1, geom, geom.Domain());
            const amrex::Real poff_z = GetPosOffset(2, geom, geom.Domain());
            if (fd.m_use_custom_size_lo) {
                cut_domain.setSmall({
                    static_cast<int>(std::round((fd.m_diag_lo[0] - poff_x)/geom.CellSize(0))),
                    static_cast<int>(std::round((fd.m_diag_lo[1] - poff_y)/geom.CellSize(1))),
                    static_cast<int>(std::round((fd.m_diag_lo[2] - poff_z)/geom.CellSize(2)))
                });
            }
            if (fd.m_use_custom_size_hi) {
                cut_domain.setBig({
                    static_cast<int>(std::round((fd.m_diag_hi[0] - poff_x)/geom.CellSize(0))),
                    static_cast<int>(std::round((fd.m_diag_hi[1] - poff_y)/geom.CellSize(1))),
                    static_cast<int>(std::round((fd.m_diag_hi[2] - poff_z)/geom.CellSize(2)))
                });
            }
            // sometimes the cut_domain is off by one cell due to rounding errors
            if (!(domain & cut_domain).ok()) {
                cut_domain.grow(1);
            }
            // calculate intersection of boxes to prevent them getting larger
            domain &= cut_domain;
        }

        amrex::RealBox diag_domain = geom.ProbDomain();
        for(int dir=0; dir<=2; ++dir) {
            // make diag_domain correspond to box
            diag_domain.setLo(dir, geom.ProbLo(dir)
                + (domain.smallEnd(dir) - geom.Domain().smallEnd(dir)) * geom.CellSize(dir));
            diag_domain.setHi(dir, geom.ProbHi(dir)
                + (domain.bigEnd(dir) - geom.Domain().bigEnd(dir)) * geom.CellSize(dir));
        }

        // trim the 3D box to slice box for slice IO
        for(int dir=0; dir<=2; ++dir) {
            if (fd.m_remove_axis[dir]) {
                const amrex::Real half_cell_size = diag_domain.length(dir) /
                                                   ( 2. * domain.length(dir) );
                const amrex::Real mid = (diag_domain.lo(dir) + diag_domain.hi(dir)) / 2.;
                // Flatten the box down to 1 cell in the approprate direction.
                domain.setSmall(dir, 0);
                domain.setBig(dir, 0);
                if ((dir != 2 || !fd.m_integrate_along_z) &&
                    fd.m_base_diag_type != DiagnosticData::diag_type::histogram)
                {
                    diag_domain.setLo(dir, mid - half_cell_size);
                    diag_domain.setHi(dir, mid + half_cell_size);
                }
            }
        }

        domain.coarsen(fd.m_diag_coarsen);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(domain.ok(),
            "Box for diagnostic object '" + fd.m_diag_name + "' is empty. "
            "Make sure that it intersects with the simulation domain!\n"
            "Simulation: " + amrex::ToString(sim_domain) + "\n"
            "Diagnostic: " + amrex::ToString(cut_domain) + "\n"
            "Intersection: " + amrex::ToString(domain)
        );

        if (fd.m_has_output) {
            HIPACE_PROFILE("Diagnostic::InitDiagnosticsStep()");

            fd.m_realspace_geom = amrex::Geometry(domain, &diag_domain, geom.Coord());

            switch (fd.m_base_diag_type) {
                case DiagnosticData::diag_type::field:
                    // real data
                    fd.m_geom_io = fd.m_realspace_geom;
                    fd.m_F_real.resize(domain, fd.m_nfields, amrex::The_Pinned_Arena());
                    fd.m_F_real.setVal<amrex::RunOn::Host>(0);
                    break;
                case DiagnosticData::diag_type::laser:
                    // complex data
                    fd.m_geom_io = fd.m_realspace_geom;
                    fd.m_F_complex.resize(domain, fd.m_nfields, amrex::The_Pinned_Arena());
                    fd.m_F_complex.setVal<amrex::RunOn::Host>({0,0});
                    break;
                case DiagnosticData::diag_type::histogram: {
                    // real data with one slice used as cache on the GPU
                    amrex::Box hist_domain = domain;
                    amrex::RealBox hist_bounds = diag_domain;
                    hist_domain.setRange(0, 0, fd.m_hist_num_bins[0]);
                    hist_bounds.setLo(0, fd.m_hist_bins_lo[0]);
                    hist_bounds.setHi(0, fd.m_hist_bins_hi[0]);
                    if (fd.m_hist_num_dims == 1) {
                        hist_domain.setRange(1, 0, 1);
                        hist_bounds.setLo(1, amrex::Real(0));
                        hist_bounds.setHi(1, amrex::Real(1));
                    } else {
                        hist_domain.setRange(1, 0, fd.m_hist_num_bins[1]);
                        hist_bounds.setLo(1, fd.m_hist_bins_lo[1]);
                        hist_bounds.setHi(1, fd.m_hist_bins_hi[1]);
                    }

                    fd.m_geom_io = amrex::Geometry(hist_domain, &hist_bounds, geom.Coord());
                    fd.m_F_real.resize(hist_domain, fd.m_nfields, amrex::The_Pinned_Arena());
                    fd.m_F_real.setVal<amrex::RunOn::Host>(0);
                    hist_domain.setRange(2, 0, 1);
                    fd.m_hist_gpu_fab.resize(hist_domain, 1, amrex::The_Arena());
                    fd.m_hist_gpu_fab.setVal<amrex::RunOn::Device>(0);
                }
                break;
                default:
                    break;
            }
        }
    }

    // all diagnostics that output particles

    for (auto& fd : m_diag_data) {

        if (!(fd.m_base_diag_type == DiagnosticData::diag_type::beam ||
            fd.m_base_diag_type == DiagnosticData::diag_type::plasma_slice ||
            fd.m_base_diag_type == DiagnosticData::diag_type::particle_boundary))
        {
            continue;
        }

        fd.m_has_output = hasOutput(fd, output_step, output_time, is_last_step);

        if (fd.m_has_output) {
            HIPACE_PROFILE("Diagnostic::InitDiagnosticsStep()");

            const std::size_t num_species = fd.m_species_names.size();

            fd.m_species_data.resize(num_species);
            fd.m_idcpu_name.resize(num_species);
            fd.m_real_names.resize(num_species);
            fd.m_int_names.resize(num_species);

            uint64_t np_total = 0;

            for (std::size_t i = 0; i < num_species; ++i) {
                const std::string& species_name = fd.m_species_names[i];
                if (plasmas.HasPlasma(species_name)) {
                    auto& plasma = plasmas.GetPlasma(species_name);
                    np_total = plasma.TotalNumberOfParticles(false, true);

                    fd.m_idcpu_name[i] = "id";

                    if (fd.m_base_diag_type == DiagnosticData::diag_type::particle_boundary) {
                        fd.m_real_names[i] = {
                            "position/x", "position/y", "position/z",
                            "weighting",
                            "momentum/x", "momentum/y", "momentum/z"
                        };
                        fd.m_int_names[i] = {};
                    } else if (fd.m_base_diag_type == DiagnosticData::diag_type::plasma_slice) {
                        fd.m_real_names[i] = plasma.GetRealSoANames();
                        fd.m_int_names[i] = plasma.GetIntSoANames();
                    }
                } else {
                    auto& beam = beams.getBeam(species_name);
                    np_total = beam.getTotalNumParticles();
                    int output_ratio = fd.m_output_ratio;
                    if (output_ratio == 1) {
                        output_ratio = beam.m_output_ratio;
                    }
                    if (output_ratio > 1) {
                        np_total = (np_total + output_ratio - 1) / output_ratio;
                    }

                    fd.m_idcpu_name[i] = "id";
                    fd.m_real_names[i] = {
                        "position/x", "position/y", "position/z",
                        "weighting",
                        "momentum/x", "momentum/y", "momentum/z"
                    };
                    if (beam.m_do_spin_tracking &&
                        fd.m_base_diag_type == DiagnosticData::diag_type::beam)
                    {
                        fd.m_real_names[i].push_back("spin/x");
                        fd.m_real_names[i].push_back("spin/y");
                        fd.m_real_names[i].push_back("spin/z");
                    }
                    fd.m_int_names[i] = {};
                }

                fd.m_species_data[i].define(
                    fd.m_real_names[i].size(),
                    fd.m_int_names[i].size(),
                    &fd.m_real_names[i],
                    &fd.m_int_names[i],
                    amrex::The_Pinned_Arena()
                );

                fd.m_species_data[i].resize(0);

                if (fd.m_base_diag_type == DiagnosticData::diag_type::beam) {
                    fd.m_species_data[i].reserve(np_total, amrex::GrowthStrategy::Exact);
                }
            }
        }
    }
}

std::pair<bool, int>
Diagnostic::ReverseShapeFactor (const DiagnosticData& fd, int islice,
                                const amrex::Geometry& geom3d)
{
    const amrex::Real poff_calc_z = GetPosOffset(2, geom3d, geom3d.Domain());
    const amrex::Real poff_diag_z = GetPosOffset(2, fd.m_realspace_geom,
                                                 fd.m_realspace_geom.Domain());

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geom3d.CellSize(2) <= fd.m_realspace_geom.CellSize(2),
        "Diagnostic cannot have a smaller cellsize in z than the simulation domain"
    );

    const amrex::Real z_pos = amrex::Real(islice) * geom3d.CellSize(2) + poff_calc_z;

    if (fd.m_integrate_along_z) {
        // integral along z
        return {
            fd.m_realspace_geom.ProbLo(2) <= z_pos && z_pos <= fd.m_realspace_geom.ProbHi(2),
            fd.m_realspace_geom.Domain().smallEnd(2)
        };
    } else {
        // zeroth order interpolation from simulation domain to diag
        const int diag_slice = static_cast<int>(std::round(
            (z_pos - poff_diag_z) * fd.m_realspace_geom.InvCellSize(2)));

        const int calc_slice = static_cast<int>(std::round(
            ((amrex::Real(diag_slice) * fd.m_realspace_geom.CellSize(2) + poff_diag_z)
            - poff_calc_z) * geom3d.InvCellSize(2)));

        return {
            calc_slice == islice &&
            fd.m_realspace_geom.Domain().smallEnd(2) <= diag_slice &&
            diag_slice <= fd.m_realspace_geom.Domain().bigEnd(2),
            diag_slice
        };
    }
}

void
Diagnostic::HistogramDepositionCopy (DiagnosticData& fd, int islice,
                                     MultiPlasma& plasmas, MultiBeam& beams,
                                     const amrex::Vector<amrex::Geometry>& field_geom)
{
    auto [collect_data, dst_slice] = ReverseShapeFactor(fd, islice, field_geom[0]);
    if (!collect_data) {
        return;
    }
    HIPACE_PROFILE("Diagnostic::HistogramDepositionCopy()");
    for (int icomp = 0; icomp < fd.m_hist_species_names.size(); ++icomp) {
        const auto& species_name = fd.m_hist_species_names[icomp];
        amrex::Real* gpu_ptr = fd.m_hist_gpu_fab.dataPtr();
        amrex::Real* cpu_ptr = fd.m_F_real.dataPtr(icomp) +
            fd.m_hist_gpu_fab.numPts() * (dst_slice - fd.m_F_real.box().smallEnd(2));
        if (fd.m_integrate_along_z) {
            // add to previous data
            amrex::Gpu::htod_memcpy_async(gpu_ptr, cpu_ptr,
                sizeof(amrex::Real) * fd.m_hist_gpu_fab.size()
            );
        } else {
            // start from zero
            fd.m_hist_gpu_fab.setVal<amrex::RunOn::Device>(0);
        }
        if (plasmas.HasPlasma(species_name)) {
            const amrex::Real zmid = amrex::Real(islice) * field_geom[0].CellSize(2) +
                GetPosOffset(2, field_geom[0], field_geom[0].Domain());
            HistogramDepositionPlasma(plasmas.GetPlasma(species_name), fd, zmid);
        } else {
            HistogramDepositionBeam(beams.getBeam(species_name), fd);
        }
        amrex::Gpu::dtoh_memcpy_async(cpu_ptr, gpu_ptr,
            sizeof(amrex::Real) * fd.m_hist_gpu_fab.size()
        );
    }
}

void
Diagnostic::CopyBeams (DiagnosticData& fd, MultiBeam& beams)
{
    HIPACE_PROFILE("Diagnostic::CopyBeams()");

    for (amrex::Long i = 0; i < fd.m_species_names.size(); ++i) {
        const std::string& species_name = fd.m_species_names[i];
        auto& beam = beams.getBeam(species_name);

        uint64_t np = beam.getNumParticles(WhichBeamSlice::This);

        int output_ratio = fd.m_output_ratio;
        if (fd.m_output_ratio == 1) {
            output_ratio = beam.m_output_ratio;
        }

        if (output_ratio > 1) {
            np = amrex::partitionParticles(beam.getBeamSlice(WhichBeamSlice::This),
                [=] AMREX_GPU_DEVICE (auto& ptd, int ip) {
                    return ip < int(np) && ptd.idcpu(ip) % output_ratio == 0;
                }
            );
        }

        if (np != 0) {
            // copy data from GPU to IO buffer
            auto& slice = beam.getBeamSlice(WhichBeamSlice::This);

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                fd.m_species_data[i].NumRealComps() <= slice.NumRealComps() &&
                fd.m_species_data[i].NumIntComps() <= slice.NumIntComps(),
                "List of real names in particle diagnostic does not match the beam");

            const auto old_size = fd.m_species_data[i].numParticles();
            const auto new_size = old_size + np;
            fd.m_species_data[i].resize(new_size, amrex::GrowthStrategy::Geometric);

            if (fd.m_idcpu_name[i] != "") {
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    slice.GetIdCPUData().begin(),
                    slice.GetIdCPUData().begin() + np,
                    fd.m_species_data[i].GetIdCPUData().data() + old_size);
            }

            for (std::size_t idx=0; idx<fd.m_real_names[i].size(); idx++) {
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    slice.GetRealData(idx).begin(),
                    slice.GetRealData(idx).begin() + np,
                    fd.m_species_data[i].GetRealData(idx).data() + old_size);
            }

            for (std::size_t idx=0; idx<fd.m_int_names[i].size(); idx++) {
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    slice.GetIntData(idx).begin(),
                    slice.GetIntData(idx).begin() + np,
                    fd.m_species_data[i].GetIntData(idx).begin() + old_size);
            }
        }
    }
}

void
Diagnostic::CopyPlasmas (DiagnosticData& fd, MultiPlasma& plasmas)
{
    HIPACE_PROFILE("Diagnostic::CopyPlasmas()");

    for (amrex::Long i = 0; i < fd.m_species_names.size(); ++i) {
        const std::string& species_name = fd.m_species_names[i];
        auto& plasma = plasmas.GetPlasma(species_name);

        for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti) {

            amrex::removeInvalidParticles(pti.GetParticleTile());
            uint64_t np = pti.numParticles();

            if (np == 0) {
                continue;
            }

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                fd.m_species_data[i].NumRealComps() == pti.GetParticleTile().NumRealComps() &&
                fd.m_species_data[i].NumIntComps() == pti.GetParticleTile().NumIntComps(),
                "List of real names in particle diagnostic does not match the beam");

            const auto old_size = fd.m_species_data[i].numParticles();
            const auto new_size = old_size + np;
            // only one chunk of particles is expected per diagnostic
            fd.m_species_data[i].resize(new_size, amrex::GrowthStrategy::Exact);

            if (fd.m_idcpu_name[i] != "") {
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    pti.GetParticleTile().GetIdCPUData().begin(),
                    pti.GetParticleTile().GetIdCPUData().begin() + np,
                    fd.m_species_data[i].GetIdCPUData().data() + old_size);
            }

            for (std::size_t idx=0; idx<fd.m_real_names[i].size(); idx++) {
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    pti.GetParticleTile().GetRealData(idx).begin(),
                    pti.GetParticleTile().GetRealData(idx).begin() + np,
                    fd.m_species_data[i].GetRealData(idx).data() + old_size);
            }

            for (std::size_t idx=0; idx<fd.m_int_names[i].size(); idx++) {
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    pti.GetParticleTile().GetIntData(idx).begin(),
                    pti.GetParticleTile().GetIntData(idx).begin() + np,
                    fd.m_species_data[i].GetIntData(idx).begin() + old_size);
            }
        }
    }
}

void
Diagnostic::CopyParticlesBoundary (DiagnosticData& fd, int islice, MultiPlasma& plasmas,
                                   MultiBeam& beams, const amrex::Vector<amrex::Geometry>& gm)
{
    HIPACE_PROFILE("Diagnostic::CopyParticlesBoundary()");

    for (amrex::Long i = 0; i < fd.m_species_names.size(); ++i) {
        const std::string& species_name = fd.m_species_names[i];

        if (plasmas.HasPlasma(species_name)) {
            auto& plasma = plasmas.GetPlasma(species_name);

            for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti) {

                uint64_t np_left = amrex::partitionParticles(pti.GetParticleTile(),
                    [=] AMREX_GPU_DEVICE (auto& ptd, int ip) {
                        return ptd.id(ip) != PlasmaID::invalid_at_boundary;
                    }
                );

                uint64_t np = pti.numParticles() - np_left;

                if (np == 0) {
                    continue;
                }

                const auto old_size = fd.m_species_data[i].numParticles();
                const auto new_size = old_size + np;
                fd.m_species_data[i].resize(new_size, amrex::GrowthStrategy::Geometric);

                auto ptd_plasma = pti.GetParticleTile().getParticleTileData();
                auto ptd_diag = fd.m_species_data[i].getParticleTileData();
                const amrex::Real plasma_z = gm[0].ProbLo(2) +
                    (islice + amrex::Real(1) - gm[0].Domain().smallEnd(2))*gm[0].CellSize(2);
                const amrex::Real dzeta_inv = gm[0].InvCellSize(2);
                const amrex::Real dt = Hipace::GetInstance().m_dt;
                const amrex::Real weight_factor = dt * get_phys_const().c * dzeta_inv;

                amrex::ParallelFor(np,
                    [=] AMREX_GPU_DEVICE (uint64_t ip) {
                        ptd_diag.idcpu(ip + old_size) = ptd_plasma.idcpu(ip + np_left);
                        ptd_diag.pos(0, ip + old_size) = ptd_plasma.pos(0, ip + np_left);
                        ptd_diag.pos(1, ip + old_size) = ptd_plasma.pos(1, ip + np_left);
                        ptd_diag.pos(2, ip + old_size) = plasma_z;
                        const amrex::Real ux = ptd_plasma.rdata(PlasmaIdx::ux)[ip + np_left];
                        const amrex::Real uy = ptd_plasma.rdata(PlasmaIdx::uy)[ip + np_left];
                        const amrex::Real psi = ptd_plasma.rdata(PlasmaIdx::psi)[ip + np_left];
                        const amrex::Real psi_inv = 1 / psi;
                        const amrex::Real gamma = plasma_gamma(ux, uy, psi, psi_inv, 0);
                        const amrex::Real uz = plasma_uz(gamma, psi);
                        ptd_diag.rdata(3)[ip + old_size] =
                            ptd_plasma.rdata(PlasmaIdx::w)[ip + np_left] * weight_factor;
                        ptd_diag.rdata(4)[ip + old_size] = ux;
                        ptd_diag.rdata(5)[ip + old_size] = uy;
                        ptd_diag.rdata(6)[ip + old_size] = uz;
                    }
                );
            }
        } else {
            auto& beam = beams.getBeam(species_name);

            uint64_t np_left = amrex::partitionParticles(beam.getBeamSlice(WhichBeamSlice::This),
                [=] AMREX_GPU_DEVICE (auto& ptd, int ip) {
                    return ptd.id(ip) != PlasmaID::invalid_at_boundary;
                }
            );

            uint64_t np = beam.getNumParticlesIncludingSlipped(WhichBeamSlice::This) - np_left;

            if (np == 0) {
                continue;
            }

            const auto old_size = fd.m_species_data[i].numParticles();
            const auto new_size = old_size + np;
            fd.m_species_data[i].resize(new_size, amrex::GrowthStrategy::Geometric);

            auto ptd_beam = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();
            auto ptd_diag = fd.m_species_data[i].getParticleTileData();

            amrex::ParallelFor(np,
                [=] AMREX_GPU_DEVICE (uint64_t ip) {
                    ptd_diag.idcpu(ip + old_size) = ptd_beam.idcpu(ip + np_left);
                    ptd_diag.pos(0, ip + old_size) = ptd_beam.pos(0, ip + np_left);
                    ptd_diag.pos(1, ip + old_size) = ptd_beam.pos(1, ip + np_left);
                    ptd_diag.pos(2, ip + old_size) = ptd_beam.pos(2, ip + np_left);
                    ptd_diag.rdata(3)[ip + old_size] = ptd_beam.rdata(BeamIdx::w)[ip + np_left];
                    ptd_diag.rdata(4)[ip + old_size] = ptd_beam.rdata(BeamIdx::ux)[ip + np_left];
                    ptd_diag.rdata(5)[ip + old_size] = ptd_beam.rdata(BeamIdx::uy)[ip + np_left];
                    ptd_diag.rdata(6)[ip + old_size] = ptd_beam.rdata(BeamIdx::uz)[ip + np_left];
                }
            );
        }
    }
}

void
Diagnostic::FillDiagnostics (int islice, int current_N_level,
                             Fields& fields, MultiLaser& lasers,
                             MultiPlasma& plasmas, MultiBeam& beams,
                             const amrex::Vector<amrex::Geometry>& field_geom)
{
    for (auto& fd : m_diag_data) {
        if (!fd.m_has_output) {
            continue;
        }
        switch (fd.m_base_diag_type) {
            case DiagnosticData::diag_type::field:
            case DiagnosticData::diag_type::laser:
                fields.Copy(current_N_level, islice, fd, field_geom, lasers);
                break;
            case DiagnosticData::diag_type::histogram:
                if (!fd.m_hist_exit_boundary) {
                    HistogramDepositionCopy(fd, islice, plasmas, beams, field_geom);
                }
                break;
            case DiagnosticData::diag_type::beam:
                CopyBeams(fd, beams);
                break;
            case DiagnosticData::diag_type::plasma_slice:
                if (islice == fd.m_plasma_output_slice) {
                    CopyPlasmas(fd, plasmas);
                }
                break;
            case DiagnosticData::diag_type::particle_boundary:
                break;
        }
    }
}

void
Diagnostic::FillBoundaryDiagnostics (int islice, MultiPlasma& plasmas, MultiBeam& beams,
                                     const amrex::Vector<amrex::Geometry>& field_geom)
{
    // particles where already pushed so they are on the next slice now
    if (islice == 0) {
        return;
    }

    std::set<std::string> species_names_with_boundary_diag;

    // first do all diags before removing the boundary id tag
    for (auto& fd : m_diag_data) {
        if (!fd.m_has_output) {
            continue;
        }

        if (fd.m_base_diag_type == DiagnosticData::diag_type::histogram &&
            fd.m_hist_exit_boundary)
        {
            HistogramDepositionCopy(fd, islice - 1, plasmas, beams, field_geom);
            species_names_with_boundary_diag.insert(
                fd.m_hist_species_names.begin(), fd.m_hist_species_names.end());
        }

        if (fd.m_base_diag_type == DiagnosticData::diag_type::particle_boundary) {
            CopyParticlesBoundary(fd, islice - 1, plasmas, beams, field_geom);
            species_names_with_boundary_diag.insert(
                fd.m_species_names.begin(), fd.m_species_names.end());
        }
    }

    // reset id so we don't double count particles.
    for (const auto& species_name : species_names_with_boundary_diag) {
        if (plasmas.HasPlasma(species_name)) {
            auto& plasma = plasmas.GetPlasma(species_name);
            for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
            {
                const auto ptd = pti.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(
                    pti.numParticles(),
                    [=] AMREX_GPU_DEVICE (int ip) {
                        if (ptd.id(ip) == PlasmaID::invalid_at_boundary) {
                            ptd.id(ip) = PlasmaID::invalid;
                        }
                    });
            }
        } else {
            auto& beam = beams.getBeam(species_name);
            const auto ptd = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();
            amrex::ParallelFor(
                beam.getNumParticles(WhichBeamSlice::This),
                [=] AMREX_GPU_DEVICE (int ip) {
                    if (ptd.id(ip) == PlasmaID::invalid_at_boundary) {
                        ptd.id(ip) = PlasmaID::invalid;
                    }
                });
        }
    }
}

void
Diagnostic::WriteDiagnostics (
    const MultiLaser& multi_laser, const MultiBeam& beams, const MultiPlasma& plasmas,
    const amrex::Geometry& geom, const amrex::Real physical_time, const int output_step,
    amrex::Real output_time, bool is_last_step
)
{
#ifdef HIPACE_USE_OPENPMD
    if (hasAnyOutput(output_step, output_time, is_last_step)) {
        m_openpmd_writer.WriteDiagnostics(m_diag_data, multi_laser, beams, plasmas, geom,
            physical_time, output_step);
    }
#else
    amrex::ignore_unused(multi_laser, beams, plasmas, geom, physical_time, output_step,
        output_time, is_last_step);
#endif
}
