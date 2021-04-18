#include "FieldDiagnostic.H"
#include "fields/Fields.H"
#include "Hipace.H"
#include <AMReX_ParmParse.H>

FieldDiagnostic::FieldDiagnostic (int nlev)
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
            {"ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz", "jx_beam", "jy_beam",
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
}

void
FieldDiagnostic::AllocData (int lev, const amrex::Box& bx, int nfields, amrex::Geometry const& geom)
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
FieldDiagnostic::ResizeFDiagFAB (const amrex::Box box, const int lev)
{
    amrex::Box io_box = TrimIOBox(box);
    m_F[lev].resize(io_box, m_nfields);
 }

amrex::Box
FieldDiagnostic::TrimIOBox (const amrex::Box box_3d)
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
