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

    std::map<int,amrex::Vector<amrex::Box> > boxes;
    for (int i = 0; i < ba.size(); ++i) {
        int rank = dm[i];
        if (m_hipace->InSameTransverseCommunicator(rank)) {
            boxes[rank].push_back(ba[i]);
        }
    }

    // We assume each process may have multiple Boxes longitude direction, but only one Box in the
    // transverse direction.  The union of all Boxes on a process is rectangular.  The slice
    // BoxArray therefore has one Box per process.  The Boxes in the slice BoxArray have one cell in
    // the longitude direction.  We will use the lowest longitude index in each process to construct
    // the Boxes.  These Boxes do not have any overlaps. Transversely, there are no gaps between
    // them.

    amrex::BoxList bl;
    amrex::Vector<int> procmap;
    for (auto const& kv : boxes) {
        int const iproc = kv.first;
        auto const& boxes_i = kv.second;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(boxes_i.size() > 0,
                                         "We assume each process has at least one Box");
        amrex::Box bx = boxes_i[0];
        for (int j = 1; j < boxes_i.size(); ++j) {
            amrex::Box const& bxj = boxes_i[j];
            for (int idim = 0; idim < Direction::z; ++idim) {
                AMREX_ALWAYS_ASSERT(bxj.smallEnd(idim) == bx.smallEnd(idim));
                AMREX_ALWAYS_ASSERT(bxj.bigEnd(idim) == bx.bigEnd(idim));
                if (bxj.smallEnd(Direction::z) < bx.smallEnd(Direction::z)) {
                    bx = bxj;
                }
            }
        }
        bx.setBig(Direction::z, bx.smallEnd(Direction::z));
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

void
Fields::TransverseDerivative (const amrex::MultiFab& src, amrex::MultiFab& dst, const int direction,
                              const amrex::Real dx, const int scomp, const int dcomp)
{
    AMREX_ALWAYS_ASSERT((direction == Direction::x) || (direction == Direction::y));
    for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (direction == Direction::x){
                    /* finite difference along x */
                    dst_array(i,j,k,dcomp) =
                        (src_array(i+1, j, k, scomp) - src_array(i-1, j, k, scomp)) / (2*dx);
                } else /* Direction::y */ {
                    /* finite difference along y */
                    dst_array(i,j,k,dcomp) =
                        (src_array(i, j+1, k, scomp) - src_array(i, j-1, k, scomp)) / (2*dx);
                }
            }
            );
    }
}

void
Fields::Copy (int lev, int i_slice, FieldCopyType copy_type, int slice_comp, int full_comp,
              int ncomp)
{
    auto& slice_mf = m_slices[lev][1];  // always slice #1
    amrex::Array4<amrex::Real> slice_array; // There is only one Box.
    for (amrex::MFIter mfi(slice_mf); mfi.isValid(); ++mfi) {
        auto& slice_fab = slice_mf[mfi];
        amrex::Box slice_box = slice_fab.box();
        slice_box.setSmall(Direction::z, i_slice);
        slice_box.setBig  (Direction::z, i_slice);
        slice_array = amrex::makeArray4(slice_fab.dataPtr(), slice_box, slice_fab.nComp());
        // slice_array's longitude index is i_slice.
    }

    auto& full_mf = m_F[lev];
    for (amrex::MFIter mfi(full_mf); mfi.isValid(); ++mfi) {
        amrex::Box const& vbx = mfi.validbox();
        if (vbx.smallEnd(Direction::z) <= i_slice and
            vbx.bigEnd  (Direction::z) >= i_slice)
        {
            amrex::Box copy_box = amrex::grow(vbx, m_slices_nguards);
            copy_box.setSmall(Direction::z, i_slice);
            copy_box.setBig  (Direction::z, i_slice);
            auto const& full_array = full_mf.array(mfi);
            if (copy_type == FieldCopyType::FtoS) {
                amrex::ParallelFor(copy_box, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    slice_array(i,j,k,n+slice_comp) = full_array(i,j,k,n+full_comp);
                });
            } else {
                amrex::ParallelFor(copy_box, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    full_array(i,j,k,n+full_comp) = slice_array(i,j,k,n+slice_comp);
                });
            }
        }
    }
}

void
Fields::ShiftSlices (int lev)
{
    std::swap(m_slices[lev][2], m_slices[lev][3]);
    std::swap(m_slices[lev][1], m_slices[lev][2]);
}
