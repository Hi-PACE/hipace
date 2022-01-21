#include "Diagnostic.H"
#include "Hipace.H"
#include <AMReX_ParmParse.H>

Diagnostic::Diagnostic (int nlev)
    : m_F(nlev),
      m_diag_coarsen(nlev),
      m_geom_io(nlev),
      m_is_active(nlev)
{
    amrex::ParmParse ppd("diagnostic");
    std::string str_type;
    getWithParser(ppd, "diag_type", str_type);
    if        (str_type == "xyz"){
        m_diag_type = DiagType::xyz;
        m_slice_dir = -1;
    } else if (str_type == "xz") {
        m_diag_type = DiagType::xz;
        m_slice_dir = 1;
    } else if (str_type == "yz") {
        m_diag_type = DiagType::yz;
        m_slice_dir = 0;
    } else {
        amrex::Abort("Unknown diagnostics type: must be xyz, xz or yz.");
    }

    queryWithParser(ppd, "include_ghost_cells", m_include_ghost_cells);

    for(int ilev = 0; ilev<nlev; ++ilev) {
        amrex::Array<int,3> diag_coarsen_arr{1,1,1};
        // set all levels the same for now
        queryWithParser(ppd, "coarsening", diag_coarsen_arr);
        if(m_slice_dir == 0 || m_slice_dir == 1) {
            diag_coarsen_arr[m_slice_dir] = 1;
        }
        m_diag_coarsen[ilev] = amrex::IntVect(diag_coarsen_arr);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_diag_coarsen[ilev].min() >= 1,
            "Coarsening ratio must be >= 1");
    }

    queryWithParser(ppd, "field_data", m_comps_output);
    const amrex::Vector<std::string> all_field_comps
            {"ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jx_beam", "jy", "jy_beam", "jz",
             "jz_beam", "rho", "Psi"};
    if(m_comps_output.empty()) {
        m_comps_output = all_field_comps;
    }
    else {
        for(std::string comp_name : m_comps_output) {
            if(comp_name == "all" || comp_name == "All") {
                m_comps_output = all_field_comps;
                break;
            }
            if(comp_name == "none" || comp_name == "None") {
                m_comps_output.clear();
                break;
            }
            if(Comps[WhichSlice::This].count(comp_name) == 0 || comp_name == "N") {
                amrex::Abort("Unknown field diagnostics component: " + comp_name + "\nmust be " +
                "'all', 'none' or a subset of: ExmBy EypBx Ez Bx By Bz jx jy jz jx_beam jy_beam " +
                "jz_beam rho Psi" );
            }
        }
    }
    m_nfields = m_comps_output.size();
    m_comps_output_idx = amrex::Gpu::DeviceVector<int>(m_nfields);
    for(int i = 0; i < m_nfields; ++i) {
        m_comps_output_idx[i] = Comps[WhichSlice::This][m_comps_output[i]];
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
}

void
Diagnostic::AllocData (int lev)
{
    // only usable after ResizeFDiagFAB
    amrex::Box dummy_bx = {{0,0,0}, {0,0,0}};
    m_F.push_back(amrex::FArrayBox(dummy_bx, m_nfields, amrex::The_Pinned_Arena()));
}

void
Diagnostic::ResizeFDiagFAB (amrex::Box local_box, amrex::Box domain, const int lev, amrex::Geometry const& geom)
{
    if (m_include_ghost_cells) {
        local_box.grow(Fields::m_slices_nguards);
        domain.grow(Fields::m_slices_nguards);
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
    TrimIOBox(local_box, domain, diag_domain);

    local_box.coarsen(m_diag_coarsen[lev]);
    domain.coarsen(m_diag_coarsen[lev]);

    m_geom_io[lev] = amrex::Geometry(domain, &diag_domain, geom.Coord());

    std::cout << " Diag Domain: " << m_geom_io[lev].Domain().smallEnd() << " " << m_geom_io[lev].Domain().bigEnd() << std::endl;

    m_is_active[lev] = local_box.ok();

    if(m_is_active[lev]) {
        m_F[lev].resize(local_box, m_nfields);
        m_F[lev].setVal<amrex::RunOn::Device>(0);
    }
}

void
Diagnostic::TrimIOBox (amrex::Box& box_3d, amrex::Box& domain_3d, amrex::RealBox& rbox_3d)
{
    if (m_slice_dir >= 0){
        // Flatten the box down to 1 cell in the approprate direction.
        box_3d.setSmall(m_slice_dir, 0);
        box_3d.setBig  (m_slice_dir, 0);
        domain_3d.setSmall(m_slice_dir, 0);
        domain_3d.setBig  (m_slice_dir, 0);
        const amrex::Real mid = (rbox_3d.lo(m_slice_dir) + rbox_3d.hi(m_slice_dir))/2;
        rbox_3d.setLo(m_slice_dir, mid);
        rbox_3d.setHi(m_slice_dir, mid);
    }
}
