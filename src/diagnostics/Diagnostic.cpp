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
Diagnostic::ReadParameters (int nlev, bool use_laser)
{
    amrex::ParmParse ppd("diagnostic");
    amrex::ParmParse pph("hipace");

    // Make the default diagnostic objects, subset of: lev0, lev1, lev2, laser_diag
    amrex::Vector<std::string> field_diag_names{};
    for (int lev = 0; lev<nlev; ++lev) {
        std::string diag_name = "lev" + std::to_string(lev);
        field_diag_names.emplace_back(diag_name);
    }
    if (use_laser) {
        std::string diag_name = "laser_diag";
        field_diag_names.emplace_back(diag_name);
    }

    queryWithParser(ppd, "names", field_diag_names);
    if (field_diag_names.size() > 0 && field_diag_names[0] == "no_field_diag") {
        field_diag_names.clear();
    }

    m_diag_data.resize(field_diag_names.size());

    for(amrex::Long i = 0; i < m_diag_data.size(); ++i) {
        m_diag_data[i].m_diag_name = field_diag_names[i];
    }

    if (queryWithParser(pph, "output_period", m_beam_output_period.m_func_str)) {
        amrex::Print() << "WARNING: 'hipace.output_period' is deprecated! "
            "Use 'diagnostic.output_period' instead!\n";
    }
    queryWithParser(ppd, "output_period", m_beam_output_period.m_func_str);
    queryWithParser(ppd, "beam_output_period", m_beam_output_period.m_func_str);
    m_beam_output_period.compile();
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

void
Diagnostic::Initialize (int nlev, bool use_laser) {
    amrex::ParmParse ppd("diagnostic");
    amrex::ParmParse pph("hipace");

    // for each diagnostic object, choose a geometry and assign field_data

    // for the default diagnostics, what is the default geometry
    std::map<std::string, std::string> diag_name_to_default_geometry{};
    // for each geometry name, is it based on fields or laser
    std::map<std::string, DiagnosticData::diag_type> geometry_name_to_diag_type{};
    // for each geometry name, if its for fields what MR level is it on
    std::map<std::string, int> geometry_name_to_level{};
    // for each geometry, to which index do output components map to
    std::map<std::string, std::map<std::string, int>> geometry_name_to_output_comps_map{};
    // for each geometry, what output components are available
    std::map<std::string, std::set<std::string>> geometry_name_to_output_comps{};
    // in case there is an error, generate a string with all available geometries and components
    std::stringstream all_comps_error_str{};

    for (int lev = 0; lev<nlev; ++lev) {
        std::string diag_name = "lev" + std::to_string(lev);
        std::string geom_name = "level_" + std::to_string(lev);
        diag_name_to_default_geometry.emplace(diag_name, geom_name);
        geometry_name_to_diag_type.emplace(geom_name, DiagnosticData::diag_type::field);
        geometry_name_to_level.emplace(geom_name, lev);
        geometry_name_to_output_comps_map[geom_name] = Comps[WhichSlice::This];
        // add derived diagnostics for Ex and Ey
        geometry_name_to_output_comps_map[geom_name]["Ex"] = -1;
        geometry_name_to_output_comps_map[geom_name]["Ey"] = -2;
    }
    if (use_laser) {
        std::string diag_name = "laser_diag";
        std::string geom_name = "laser";
        diag_name_to_default_geometry.emplace(diag_name, geom_name);
        geometry_name_to_diag_type.emplace(geom_name, DiagnosticData::diag_type::laser);
        geometry_name_to_output_comps_map[geom_name]["laserEnvelope"] = WhichLaserSlice::n00j00_r;
        // real=chi, imag=chi_initial
        geometry_name_to_output_comps_map[geom_name]["laserChi"] = WhichLaserSlice::chi;
        // add derived diagnostics for |a^2|
        geometry_name_to_output_comps_map[geom_name]["|a^2|"] = -1;
    }
    { // histogram
        std::string geom_name = "histogram";
        geometry_name_to_diag_type.emplace(geom_name, DiagnosticData::diag_type::histogram);
        geometry_name_to_output_comps_map[geom_name]; // insert empty map
    }

    for (const auto& [geom_name, comp_map] : geometry_name_to_output_comps_map) {
        all_comps_error_str << "Available components for  '"
            << geom_name << "':\n    ";
        for (const auto& [comp, idx] : comp_map) {
            geometry_name_to_output_comps[geom_name].insert(comp);
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

        std::string base_geom_name = "level_0";

        if (diag_name_to_default_geometry.count(fd.m_diag_name) > 0) {
            base_geom_name = diag_name_to_default_geometry.at(fd.m_diag_name);
        }

        // backward compatibility
        queryWithParserAlt(pp, "base_geometry", base_geom_name, ppd);
        DeprecatedInput(fd.m_diag_name, "level", "base_geometry");

        if (geometry_name_to_diag_type.count(base_geom_name) > 0) {
            fd.m_base_diag_type = geometry_name_to_diag_type.at(base_geom_name);
        } else {
            amrex::Abort("Unknown diagnostics base_geometry: '" + base_geom_name + "'!\n" +
                         all_comps_error_str.str());
        }

        if (fd.m_base_diag_type == DiagnosticData::diag_type::field) {
            fd.m_level = geometry_name_to_level.at(base_geom_name);
        }

        // general parametes for all base geometries

        if (queryWithParser(pph, "output_period", fd.m_output_period.m_func_str)) {
            amrex::Print() << "WARNING: 'hipace.output_period' is deprecated! "
                "Use 'diagnostic.output_period' instead!\n";
        }
        queryWithParserAlt(pp, "output_period", fd.m_output_period.m_func_str, ppd);
        fd.m_output_period.compile();

        fd.m_use_custom_size_lo = queryWithParserAlt(pp, "patch_lo", fd.m_diag_lo, ppd);
        fd.m_use_custom_size_hi = queryWithParserAlt(pp, "patch_hi", fd.m_diag_hi, ppd);

        amrex::Array<int,3> diag_coarsen_arr{1,1,1};
        queryWithParserAlt(pp, "coarsening", diag_coarsen_arr, ppd);
        fd.m_diag_coarsen = amrex::IntVect(diag_coarsen_arr);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fd.m_diag_coarsen.min() >= 1,
                    "Coarsening ratio must be >= 1");

        queryWithParserAlt(pp, "include_ghost_cells", fd.m_include_ghost_cells, ppd);

        // parameters for specific base geometries

        switch (fd.m_base_diag_type) {
            case DiagnosticData::diag_type::field:
            case DiagnosticData::diag_type::laser: {
                std::string str_type;
                getWithParserAlt(pp, "diag_type", str_type, ppd);
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

            }
            break;
            case DiagnosticData::diag_type::histogram: {
                getWithParser(pp, "hist_species_names", fd.m_hist_species_names);
                getWithParser(pp, "hist_num_bins", fd.m_hist_num_bins);
                getWithParser(pp, "hist_bins_lo", fd.m_hist_bins_lo);
                getWithParser(pp, "hist_bins_hi", fd.m_hist_bins_hi);
                bool add_z_axis = false;
                queryWithParser(pp, "hist_add_z_axis", add_z_axis);
                fd.m_integrate_along_z = !add_z_axis;

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
            break;
        }

        if (fd.m_base_diag_type == DiagnosticData::diag_type::histogram) {
            // no need to have field_data with histogram
            continue;
        }

        // get and parse field_data parameter

        amrex::Vector<std::string> use_comps{};
        const bool use_local_comps = queryWithParser(pp, "field_data", use_comps);
        if (!use_local_comps) {
            queryWithParser(ppd, "field_data", use_comps);
        }

        // set to store all used components to avoid duplicates
        std::set<std::string> comps_set{};

        if (use_comps.empty()) {
            // by default output all components
            use_comps.push_back("all");
        }

        // iterate through the user-provided components from left to right
        for (const std::string& comp_name : use_comps) {
            if (comp_name == "all" || comp_name == "All") {
                is_global_comp_used[comp_name] = true;
                // insert all available components
                comps_set.insert(geometry_name_to_output_comps[base_geom_name].begin(),
                                 geometry_name_to_output_comps[base_geom_name].end());
            } else if (comp_name == "none" || comp_name == "None") {
                is_global_comp_used[comp_name] = true;
                // remove all components
                comps_set.clear();
            } else if (geometry_name_to_output_comps[base_geom_name].count(comp_name) > 0) {
                is_global_comp_used[comp_name] = true;
                // insert requested component
                comps_set.insert(comp_name);
            } else if (comp_name.find("remove_") == 0 &&
                       geometry_name_to_output_comps[base_geom_name].count(
                       comp_name.substr(std::string("remove_").size(), comp_name.size())) > 0) {
                is_global_comp_used[comp_name] = true;
                // remove requested component
                comps_set.erase(comp_name.substr(std::string("remove_").size(), comp_name.size()));
            } else if (use_local_comps) {
                // if field_data was specified through <diag name>,
                // assert that all components exist in the geometry
                amrex::Abort("Unknown diagnostics field_data '" + comp_name +
                             "' in base_geometry '" + base_geom_name + "'!\n" +
                             all_comps_error_str.str());
            } else {
                // if field_data was specified through diagnostic,
                // check later that all components are at least used by one of the diagnostics
                is_global_comp_used.try_emplace(comp_name, false);
            }
        }

        fd.m_comps_output.assign(comps_set.begin(), comps_set.end());
        fd.m_nfields = fd.m_comps_output.size();

        // copy the indexes of m_comps_output to the GPU
        fd.m_comps_output_idx.resize(fd.m_nfields);
        for (int i = 0; i < fd.m_nfields; ++i) {
            fd.m_comps_output_idx[i] =
                geometry_name_to_output_comps_map.at(base_geom_name).at(fd.m_comps_output[i]);
        }
        fd.m_comps_output_idx.copyToDeviceAsync();
    }

    // check that all components are at least used by one of the diagnostics
    for (auto& [key, val] : is_global_comp_used) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(val,
            "Unknown or unused component in diagnostic.field_data.\n'" +
            key + "' does not belong to any diagnostic.names!\n" +
            all_comps_error_str.str()
        );
    }

    // if there are multiple diagnostic objects with the same m_base_diag_type (colliding component
    // names), append the name of the diagnostic object to the component name in the output
    for (auto& fd : m_diag_data) {
        if (fd.m_base_diag_type == DiagnosticData::diag_type::histogram ||
            1 < std::count_if(m_diag_data.begin(), m_diag_data.end(), [&] (auto& fd2) {
            return fd.m_base_diag_type == fd2.m_base_diag_type;
        })) {
            for (auto& comp_name : fd.m_comps_output) {
                comp_name += "_" + fd.m_diag_name;
            }
        }
    }

    amrex::ParmParse ppb("beams");
    // read in all beam names
    amrex::Vector<std::string> all_beam_names;
    queryWithParser(ppb, "names", all_beam_names);
    // read in which beam should be written to file
    queryWithParser(ppd, "beam_data", m_output_beam_names);

    if(m_output_beam_names.empty()) {
        m_output_beam_names = all_beam_names;
    } else {
        for(std::string beam_name : m_output_beam_names) {
            if(beam_name == "all" || beam_name == "All") {
                m_output_beam_names = all_beam_names;
                break;
            }
            if(beam_name == "none" || beam_name == "None") {
                m_output_beam_names.clear();
                break;
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::find(all_beam_names.begin(), all_beam_names.end(), beam_name)
                    != all_beam_names.end(),
                "Unknown beam name: " + beam_name + "\nmust be " +
                "a subset of beams.names: " + amrex::ToString(all_beam_names) + ", 'all' or 'none'"
            );
        }
    }
}

void
Diagnostic::ResizeFDiagFAB (amrex::Vector<amrex::Geometry>& field_geom,
                            amrex::Geometry const& laser_geom, int output_step,
                            amrex::Real output_time, bool is_last_step)
{
    for (auto& fd : m_diag_data) {

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

        fd.m_has_output = hasFieldOutput(fd, output_step, output_time, is_last_step);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(domain.ok(),
            "Box for diagnostic object '" + fd.m_diag_name + "' is empty. "
            "Make sure that it intersects with the simulation domain!\n"
            "Simulation: " + amrex::ToString(sim_domain) + "\n"
            "Diagnostic: " + amrex::ToString(cut_domain) + "\n"
            "Intersection: " + amrex::ToString(domain)
        );

        if(fd.m_has_output) {
            HIPACE_PROFILE("Diagnostic::ResizeFDiagFAB()");

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
            case DiagnosticData::diag_type::histogram: {
                auto [collect_data, dst_slice] = ReverseShapeFactor(fd, islice, field_geom[0]);
                if (!collect_data) {
                    break;
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
            break;
        }
    }
}
