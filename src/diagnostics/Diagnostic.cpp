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
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

Diagnostic::Diagnostic (int nlev, bool use_laser)
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

    m_field_data.resize(field_diag_names.size());

    for(amrex::Long i = 0; i<m_field_data.size(); ++i) {
        auto& fd = m_field_data[i];

        fd.m_diag_name = field_diag_names[i];

        amrex::ParmParse pp(fd.m_diag_name);

        std::string str_type;
        getWithParserAlt(pp, "diag_type", str_type, ppd);
        if        (str_type == "xyz"){
            fd.m_slice_dir = -1;
        } else if (str_type == "xz") {
            fd.m_slice_dir = 1;
        } else if (str_type == "yz") {
            fd.m_slice_dir = 0;
        } else if (str_type == "xy_integrated") {
            fd.m_slice_dir = 2;
        } else {
            amrex::Abort("Unknown diagnostics type: must be xyz, xz or yz.");
        }

        queryWithParserAlt(pp, "include_ghost_cells", fd.m_include_ghost_cells, ppd);

        fd.m_use_custom_size_lo = queryWithParserAlt(pp, "patch_lo", fd.m_diag_lo, ppd);
        fd.m_use_custom_size_hi = queryWithParserAlt(pp, "patch_hi", fd.m_diag_hi, ppd);

        amrex::Array<int,3> diag_coarsen_arr{1,1,1};
        queryWithParserAlt(pp, "coarsening", diag_coarsen_arr, ppd);
        if(fd.m_slice_dir == 0 || fd.m_slice_dir == 1 || fd.m_slice_dir == 2) {
            diag_coarsen_arr[fd.m_slice_dir] = 1;
        }
        fd.m_diag_coarsen = amrex::IntVect(diag_coarsen_arr);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( fd.m_diag_coarsen.min() >= 1,
            "Coarsening ratio must be >= 1");

        queryWithParser(pph, "output_period", fd.m_output_period);
        queryWithParserAlt(pp, "output_period", fd.m_output_period, ppd);
    }

    if (queryWithParser(pph, "output_period", m_beam_output_period)) {
        amrex::Print() << "WARNING: 'hipace.output_period' is deprecated! "
            "Use 'diagnostic.output_period' instead!\n";
    }
    queryWithParser(ppd, "output_period", m_beam_output_period);
    queryWithParser(ppd, "beam_output_period", m_beam_output_period);
}

bool
Diagnostic::needsRho () const {
    amrex::ParmParse ppd("diagnostic");
    for (auto& fd : m_field_data) {
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
    for (auto& fd : m_field_data) {
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

void
Diagnostic::Initialize (int nlev, bool use_laser) {
    amrex::ParmParse ppd("diagnostic");

    // for each diagnostic object, choose a geometry and assign field_data

    // for the default diagnostics, what is the default geometry
    std::map<std::string, std::string> diag_name_to_default_geometry{};
    // for each geometry name, is it based on fields or laser
    std::map<std::string, FieldDiagnosticData::geom_type> geometry_name_to_geom_type{};
    // for each geometry name, if its for fields what MR level is it on
    std::map<std::string, int> geometry_name_to_level{};
    // for each geometry, what output components are available
    std::map<std::string, std::set<std::string>> geometry_name_to_output_comps{};
    // in case there is an error, generate a string with all available geometries and components
    std::stringstream all_comps_error_str{};
    for (int lev = 0; lev<nlev; ++lev) {
        std::string diag_name = "lev" + std::to_string(lev);
        std::string geom_name = "level_" + std::to_string(lev);
        diag_name_to_default_geometry.emplace(diag_name, geom_name);
        geometry_name_to_geom_type.emplace(geom_name, FieldDiagnosticData::geom_type::field);
        geometry_name_to_level.emplace(geom_name, lev);
        all_comps_error_str << "Available components in base_geometry '" << geom_name << "':\n    ";
        for (const auto& [comp, idx] : Comps[WhichSlice::This]) {
            geometry_name_to_output_comps[geom_name].insert(comp);
            all_comps_error_str << comp << " ";
        }
        all_comps_error_str << "\n";
    }
    if (use_laser) {
        std::string diag_name = "laser_diag";
        std::string geom_name = "laser";
        std::string laser_io_name = "laserEnvelope";
        diag_name_to_default_geometry.emplace(diag_name, geom_name);
        geometry_name_to_geom_type.emplace(geom_name, FieldDiagnosticData::geom_type::laser);
        geometry_name_to_level.emplace(geom_name, 0);
        all_comps_error_str << "Available components in base_geometry '" << geom_name << "':\n    ";
        geometry_name_to_output_comps[geom_name].insert(laser_io_name);
        all_comps_error_str << laser_io_name << "\n";
    }
    all_comps_error_str << "Additionally, 'all' and 'none' are supported as field_data\n"
                        << "Components can be removed after 'all' by using 'remove_<comp name>'.\n";

    // keep track of all components from the input and later assert that they were all used
    std::map<std::string, bool> is_global_comp_used{};

    for (auto& fd : m_field_data) {
        amrex::ParmParse pp(fd.m_diag_name);

        std::string base_geom_name = "level_0";

        if (diag_name_to_default_geometry.count(fd.m_diag_name) > 0) {
            base_geom_name = diag_name_to_default_geometry.at(fd.m_diag_name);
        }

        queryWithParserAlt(pp, "base_geometry", base_geom_name, ppd);

        if (geometry_name_to_geom_type.count(base_geom_name) > 0) {
            fd.m_base_geom_type = geometry_name_to_geom_type.at(base_geom_name);
            fd.m_level = geometry_name_to_level.at(base_geom_name);
        } else {
            amrex::Abort("Unknown diagnostics base_geometry: '" + base_geom_name + "'!\n" +
                         all_comps_error_str.str());
        }

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
        if (fd.m_base_geom_type == FieldDiagnosticData::geom_type::field) {
            amrex::Gpu::PinnedVector<int> local_comps_output_idx(fd.m_nfields);
            for(int i = 0; i < fd.m_nfields; ++i) {
                local_comps_output_idx[i] = Comps[WhichSlice::This][fd.m_comps_output[i]];
            }
            fd.m_comps_output_idx.assign(local_comps_output_idx.begin(), local_comps_output_idx.end());
        }
    }

    // check that all components are at least used by one of the diagnostics
    for (auto& [key, val] : is_global_comp_used) {
        if (!val) {
            amrex::Abort("Unknown or unused component in diagnostic.field_data.\n'" +
                         key + "' does not belong to any diagnostic.names!\n" +
                         all_comps_error_str.str());
        }
    }

    // if there are multiple diagnostic objects with the same m_base_geom_type (colliding component
    // names), append the name of the diagnostic object to the component name in the output
    for (auto& fd : m_field_data) {
        if (1 < std::count_if(m_field_data.begin(), m_field_data.end(), [&] (auto& fd2) {
            return fd.m_base_geom_type == fd2.m_base_geom_type;
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
            if(std::find(all_beam_names.begin(), all_beam_names.end(), beam_name) ==  all_beam_names.end() ) {
                amrex::Abort("Unknown beam name: " + beam_name + "\nmust be " +
                "a subset of beams.names or 'none'");
            }
        }
    }

    m_initialized = true;
}

void
Diagnostic::ResizeFDiagFAB (amrex::Vector<amrex::Geometry>& field_geom,
                            amrex::Geometry const& laser_geom, int output_step, int max_step,
                            amrex::Real output_time, amrex::Real max_time)
{
    AMREX_ALWAYS_ASSERT(m_initialized);

    for (auto& fd : m_field_data) {

        amrex::Geometry geom;

        // choose the geometry of the diagnostic
        switch (fd.m_base_geom_type) {
            case FieldDiagnosticData::geom_type::field:
                geom = field_geom[fd.m_level];
                break;
            case FieldDiagnosticData::geom_type::laser:
                geom = laser_geom;
                break;
        }

        amrex::Box domain = geom.Domain();

        if (fd.m_include_ghost_cells) {
            domain.grow(Fields::m_slices_nguards);
        }

        {
            // shrink box to user specified bounds m_diag_lo and m_diag_hi (in real space)
            const amrex::Real poff_x = GetPosOffset(0, geom, geom.Domain());
            const amrex::Real poff_y = GetPosOffset(1, geom, geom.Domain());
            const amrex::Real poff_z = GetPosOffset(2, geom, geom.Domain());
            amrex::Box cut_domain = domain;
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
        TrimIOBox(fd.m_slice_dir, domain, diag_domain);

        domain.coarsen(fd.m_diag_coarsen);

        fd.m_geom_io = amrex::Geometry(domain, &diag_domain, geom.Coord());

        fd.m_has_field = domain.ok()
                         && hasFieldOutput(fd, output_step, max_step, output_time, max_time);

        if(fd.m_has_field) {
            HIPACE_PROFILE("Diagnostic::ResizeFDiagFAB()");
            switch (fd.m_base_geom_type) {
                case FieldDiagnosticData::geom_type::field:
                    fd.m_F.resize(domain, fd.m_nfields, amrex::The_Pinned_Arena());
                    fd.m_F.setVal<amrex::RunOn::Host>(0);
                    break;
                case FieldDiagnosticData::geom_type::laser:
                    fd.m_F_laser.resize(domain, fd.m_nfields, amrex::The_Pinned_Arena());
                    fd.m_F_laser.setVal<amrex::RunOn::Host>({0,0});
                    break;
            }
        }
    }
}

void
Diagnostic::TrimIOBox (int slice_dir, amrex::Box& domain_3d, amrex::RealBox& rbox_3d)
{
    if (slice_dir >= 0){
        const amrex::Real half_cell_size = rbox_3d.length(slice_dir) /
                                           ( 2. * domain_3d.length(slice_dir) );
        const amrex::Real mid = (rbox_3d.lo(slice_dir) + rbox_3d.hi(slice_dir)) / 2.;
        // Flatten the box down to 1 cell in the approprate direction.
        domain_3d.setSmall(slice_dir, 0);
        domain_3d.setBig  (slice_dir, 0);
        if (slice_dir < 2) {
            rbox_3d.setLo(slice_dir, mid - half_cell_size);
            rbox_3d.setHi(slice_dir, mid + half_cell_size);
        }
    }
}
