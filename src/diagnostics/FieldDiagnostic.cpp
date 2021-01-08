#include "FieldDiagnostic.H"

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
}

void
FieldDiagnostic::AllocData (int lev, const amrex::BoxArray& ba, int nfields, const amrex::DistributionMapping& dm, amrex::Geometry const& geom)
{
    m_nfields = nfields;
    // Create a xz slice BoxArray
    amrex::BoxList F_boxes;
    if (m_slice_dir >= 0){
        for (int i = 0; i < ba.size(); ++i){
            amrex::Box bx = ba[i];
            // Flatten the box down to 1 cell in the approprate direction.
            bx.setSmall(m_slice_dir, ba[i].length(m_slice_dir)/2);
            bx.setBig  (m_slice_dir, ba[i].length(m_slice_dir)/2);
            // Note: the MR is still cell-centered, although the data will be averaged to nodal.
            F_boxes.push_back(bx);
        }
    }
    amrex::BoxArray F_slice_ba(std::move(F_boxes));
    // m_F is defined on F_ba, the full or the slice BoxArray
    amrex::BoxArray F_ba = m_slice_dir >= 0 ? F_slice_ba : ba;
    // Only xy slices need guard cells, there is no deposition to/gather from the output array F.
    amrex::IntVect nguards_F = amrex::IntVect(0,0,0);
    // The Arena uses pinned memory.
    m_F[lev].define(F_ba, dm, m_nfields, nguards_F,
                    amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));

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
