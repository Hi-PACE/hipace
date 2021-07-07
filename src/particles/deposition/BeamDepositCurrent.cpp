#include "BeamDepositCurrent.H"
#include "particles/BeamParticleContainer.H"
#include "particles/deposition/BeamDepositCurrentInner.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_DenseBins.H>

void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Vector<amrex::Geometry> const& gm, int const lev ,const int islice,
                     const amrex::Box bx, int const offset, BeamBins& bins,
                     const bool do_beam_jx_jy_deposition, const int which_slice, int nghost)
{
    HIPACE_PROFILE("DepositCurrentSlice_BeamParticleContainer()");
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm[lev].CellSize();

    // beam deposits only up to its finest level
    if (beam.m_finest_level < lev) return;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    which_slice == WhichSlice::This || which_slice == WhichSlice::Next,
    "Current deposition can only be done in this slice (WhichSlice::This), or the next slice "
    " (WhichSlice::Next)");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Hipace::m_depos_order_z == 0,
        "Only order 0 deposition is allowed for beam per-slice deposition");

    // Assumes '2' == 'z' == 'the long dimension'.
    int islice_local = islice - bx.smallEnd(2);

    // Extract properties associated with the extent of the current box
    amrex::Box tilebox = bx;
    tilebox.grow({Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, Hipace::m_depos_order_z});

    amrex::RealBox const grid_box{tilebox, gm[lev].CellSize(), gm[lev].ProbLo()};
    amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
    amrex::Dim3 const lo = amrex::lbound(tilebox);

    // Extract the fields currents
    amrex::MultiFab& S = fields.getSlices(lev, which_slice);
    // we deposit to the beam currents, because the explicit solver
    // requires sometimes just the beam currents
    amrex::MultiFab jx_beam(S, amrex::make_alias, Comps[which_slice]["jx_beam"], 1);
    amrex::MultiFab jy_beam(S, amrex::make_alias, Comps[which_slice]["jy_beam"], 1);
    amrex::MultiFab jz_beam(S, amrex::make_alias, Comps[which_slice]["jz_beam"], 1);

    // Extract FabArray for this box (because there is currently no transverse
    // parallelization, the index we want in the slice multifab is always 0.
    // Fix later.
    amrex::FArrayBox& jxb_fab = jx_beam[0];
    amrex::FArrayBox& jyb_fab = jy_beam[0];
    amrex::FArrayBox& jzb_fab = jz_beam[0];

    amrex::Real lev_weight_fac = 1.;
    if (lev == 1 && Hipace::m_normalized_units) {
        // re-scaling the weight in normalized units to get the same charge density on lev 1
        // Not necessary in SI units, there the weight is the actual charge and not the density
        amrex::Real const * AMREX_RESTRICT dx_lev0 = gm[0].CellSize();
        lev_weight_fac = dx_lev0[0] * dx_lev0[1] * dx_lev0[2] / (dx[0] * dx[1] * dx[2]);
    }

    // For now: fix the value of the charge
    const amrex::Real q = beam.m_charge * lev_weight_fac;

    // Call deposition function in each box
    if        (Hipace::m_depos_order_xy == 0){
        doDepositionShapeN<0, 0>( beam, jxb_fab, jyb_fab, jzb_fab, dx, xyzmin, lo, q, islice_local,
                                  bins, offset, do_beam_jx_jy_deposition, which_slice, nghost);
    } else if (Hipace::m_depos_order_xy == 1){
        doDepositionShapeN<1, 0>( beam, jxb_fab, jyb_fab, jzb_fab, dx, xyzmin, lo, q, islice_local,
                                  bins, offset, do_beam_jx_jy_deposition, which_slice, nghost);
    } else if (Hipace::m_depos_order_xy == 2){
        doDepositionShapeN<2, 0>( beam, jxb_fab, jyb_fab, jzb_fab, dx, xyzmin, lo, q, islice_local,
                                  bins, offset, do_beam_jx_jy_deposition, which_slice, nghost);
    } else if (Hipace::m_depos_order_xy == 3){
        doDepositionShapeN<3, 0>( beam, jxb_fab, jyb_fab, jzb_fab, dx, xyzmin, lo, q, islice_local,
                                  bins, offset, do_beam_jx_jy_deposition, which_slice, nghost);
    } else {
        amrex::Abort("unknown deposition order");
    }

}
