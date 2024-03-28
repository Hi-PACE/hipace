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

Diagnostic::Diagnostic (int nlev)
{
    amrex::ParmParse ppd("diagnostic");
    amrex::ParmParse pph("hipace");

    amrex::Vector<std::string> field_diag_names{};

    for(int ilev = 0; ilev<nlev; ++ilev) {
        field_diag_names.emplace_back("lev"+std::to_string(ilev));
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

        for(int ilev = 0; ilev<nlev; ++ilev) {
            if (fd.m_diag_name == "lev"+std::to_string(ilev)) {
                fd.m_level = ilev;
            }
        }
        queryWithParserAlt(pp, "level", fd.m_level, ppd);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( 0 <= fd.m_level && fd.m_level < nlev,
            "Invalid diagnostic refinement level");

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
Diagnostic::Initialize (const int lev, bool do_laser) {
    if (lev!=0) return;

    amrex::ParmParse ppd("diagnostic");

    for (auto& fd : m_field_data) {
        amrex::ParmParse pp(fd.m_diag_name);

        queryWithParserAlt(pp, "field_data", fd.m_comps_output, ppd);

        amrex::Vector<std::string> all_field_comps{};
        all_field_comps.reserve(Comps[WhichSlice::This].size() + (do_laser ? 1 : 0));
        for (const auto& [comp, idx] : Comps[WhichSlice::This]) {
            all_field_comps.push_back(comp);
        }
        if (do_laser) {
            all_field_comps.push_back(fd.m_laser_io_name);
        }

        if(fd.m_comps_output.empty()) {
            fd.m_comps_output = all_field_comps;
        } else {
            for (const std::string& comp_name : fd.m_comps_output) {
                if (comp_name == "all" || comp_name == "All") {
                    fd.m_comps_output = all_field_comps;
                    break;
                }
                if (comp_name == "none" || comp_name == "None") {
                    fd.m_comps_output.clear();
                    break;
                }
                if (std::find(all_field_comps.begin(), all_field_comps.end(), comp_name) ==  all_field_comps.end()) {
                    std::stringstream error_str{};
                    error_str << "Unknown field diagnostics component: " << comp_name <<"\nmust be "
                        << "'all', 'none' or a subset of:";
                    for (auto& comp : all_field_comps) {
                        error_str << " " << comp;
                    }
                    amrex::Abort(error_str.str());
                }
            }
        }

        // remove laser from fd.m_comps_output because it is output separately
        for (auto it = fd.m_comps_output.begin(); it != fd.m_comps_output.end();) {
            if (*it == fd.m_laser_io_name) {
                it = fd.m_comps_output.erase(it);
                fd.m_do_laser = true;
            } else {
                ++it;
            }
        }

        fd.m_nfields = fd.m_comps_output.size();

        amrex::Gpu::PinnedVector<int> local_comps_output_idx(fd.m_nfields);
        for(int i = 0; i < fd.m_nfields; ++i) {
            local_comps_output_idx[i] = Comps[WhichSlice::This][fd.m_comps_output[i]];
        }
        fd.m_comps_output_idx.assign(local_comps_output_idx.begin(), local_comps_output_idx.end());

        if (m_field_data.size() != 1) {
            for (auto& comp_name : fd.m_comps_output) {
                comp_name += "_" + fd.m_diag_name;
            }
            if (fd.m_do_laser) {
                fd.m_laser_io_name += "_" + fd.m_diag_name;
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
Diagnostic::ResizeFDiagFAB (const amrex::Box a_domain, const int lev,
                            amrex::Geometry const& geom, int output_step, int max_step,
                            amrex::Real output_time, amrex::Real max_time)
{
    AMREX_ALWAYS_ASSERT(m_initialized);

    for (auto& fd : m_field_data) {

        if (fd.m_level != lev) continue;

        amrex::Box domain = a_domain;

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
            fd.m_F.resize(domain, fd.m_nfields, amrex::The_Pinned_Arena());
            fd.m_F.setVal<amrex::RunOn::Host>(0);

            if (fd.m_do_laser) {
                fd.m_F_laser.resize(domain, 1, amrex::The_Pinned_Arena());
                fd.m_F_laser.setVal<amrex::RunOn::Host>({0,0});
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
