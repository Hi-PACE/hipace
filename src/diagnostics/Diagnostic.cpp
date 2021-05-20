#include "Diagnostic.H"
#include "Hipace.H"
#include <AMReX_ParmParse.H>

Diagnostic::Diagnostic (int nlev)
    : m_F(nlev),
      m_geom_io(nlev)
{
    amrex::ParmParse ppd("diagnostic");
    std::string str_type;
    ppd.get("diag_type", str_type);
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

    ppd.queryarr("field_data", m_comps_output);
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

    amrex::ParmParse ppb("beams");
    // read in all beam names
    amrex::Vector<std::string> all_beam_names;
    ppb.queryarr("names", all_beam_names);
    // read in which beam should be written to file
    ppd.queryarr("beam_data", m_output_beam_names);

    if(m_comps_output.empty()) {
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
Diagnostic::AllocData (int lev, const amrex::Box& bx, int nfields, amrex::Geometry const& geom)
{
    m_nfields = nfields;

    // trim the 3D box to slice box for slice IO
    amrex::Box F_bx = TrimIOBox(bx);

    m_F.push_back(amrex::FArrayBox(F_bx, m_nfields, amrex::The_Pinned_Arena()));

    m_geom_io[lev] = geom;
    amrex::RealBox prob_domain = geom.ProbDomain();
    amrex::Box domain = geom.Domain();
    // Define slice box
    if (m_slice_dir >= 0){
        int const icenter = domain.length(m_slice_dir)/2;
        domain.setSmall(m_slice_dir, icenter);
        domain.setBig(m_slice_dir, icenter);
        m_geom_io[lev] = amrex::Geometry(domain, &prob_domain, geom.Coord());
    }
}

void
Diagnostic::ResizeFDiagFAB (const amrex::Box box, const int lev)
{
    amrex::Box io_box = TrimIOBox(box);
    m_F[lev].resize(io_box, m_nfields);
 }

amrex::Box
Diagnostic::TrimIOBox (const amrex::Box box_3d)
{
    // Create a xz slice Box
    amrex::Box slice_bx = box_3d;
    if (m_slice_dir >= 0){
            // Flatten the box down to 1 cell in the approprate direction.
            slice_bx.setSmall(m_slice_dir, box_3d.length(m_slice_dir)/2);
            slice_bx.setBig  (m_slice_dir, box_3d.length(m_slice_dir)/2);
    }
    // m_F is defined on F_bx, the full or the slice Box
    amrex::Box F_bx = m_slice_dir >= 0 ? slice_bx : box_3d;

    return F_bx;
}
