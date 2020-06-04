#include "Fields.H"
#include "Hipace.H"

Fields::Fields (Hipace const* a_hipace)
    : m_hipace(a_hipace),
      m_F(a_hipace->maxLevel()+1),
      m_slices(a_hipace->maxLevel()+1)
{}

void
Fields::AllocData (int lev, const amrex::BoxArray& ba,
                   const amrex::DistributionMapping& dm)
{
    m_F[lev].define(ba, dm, FieldComps::nfields, m_nguards,
                    amrex::MFInfo().SetArena(amrex::The_Arena())); // The Arena uses managed memory.

    amrex::Vector<amrex::Vector<amrex::Box> > boxes(amrex::ParallelDescriptor::NProcs());
    for (int i = 0; i < ba.size(); ++i) {
        boxes[dm[i]].push_back(ba[i]);
    }

    amrex::BoxList bl;
    amrex::Vector<int> procmap;
    for (int iproc = 0; iproc < boxes.size(); ++iproc) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(boxes[iproc].size()  >0,
                                         "We assume each process has at least one Box");
        amrex::Box bx = boxes[iproc][0];
        for (int j = 1; j < boxes[iproc].size(); ++j) {
            amrex::Box const& bxj = boxes[iproc][j];
            for (int idim = 0; idim < AMREX_SPACEDIM-1; ++idim) {
                AMREX_ALWAYS_ASSERT(bxj.smallEnd(idim) == bx.smallEnd(idim));
                AMREX_ALWAYS_ASSERT(bxj.bigEnd(idim) == bx.bigEnd(idim));
                if (bxj.smallEnd(AMREX_SPACEDIM) < bx.smallEnd(AMREX_SPACEDIM)) {
                    bx = bxj;
                }
            }
        }
        bx.setBig(AMREX_SPACEDIM, bx.smallEnd(AMREX_SPACEDIM));
        bl.push_back(bx);
        procmap.push_back(iproc);
    }

    amrex::BoxArray slice_ba(std::move(bl));
    amrex::DistributionMapping slice_dm(std::move(procmap));

    for (int islice=0; islice<m_nslices; islice++) {
        m_slices[lev][islice].define(slice_ba, slice_dm, FieldComps::nfields, m_slices_nguards,
                                     amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[lev][islice].setVal(0.0);
    }
}
