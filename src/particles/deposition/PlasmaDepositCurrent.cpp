#include "PlasmaDepositCurrent.H"

#include "particles/PlasmaParticleContainer.H"
#include "particles/deposition/PlasmaDepositCurrentInner.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields,
                const int which_slice, const bool temp_slice,
                const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                bool deposit_j_squared, amrex::Geometry const& gm, int const lev)
{
    HIPACE_PROFILE("DepositCurrent_PlasmaParticleContainer()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    which_slice == WhichSlice::This || which_slice == WhichSlice::Next ||
    which_slice == WhichSlice::RhoIons,
    "Current deposition can only be done in this slice (WhichSlice::This), the next slice "
    " (WhichSlice::Next) or for the ion charge deposition (WhichSLice::RhoIons)");

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    const amrex::Real max_qsa_weighting_factor = plasma.m_max_qsa_weighting_factor;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        amrex::Box tilebox = pti.tilebox().grow(
            {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields currents
        amrex::MultiFab& S = fields.getSlices(lev, which_slice);
        amrex::MultiFab jx(S, amrex::make_alias, Comps[which_slice]["jx"], 1);
        amrex::MultiFab jy(S, amrex::make_alias, Comps[which_slice]["jy"], 1);
        amrex::MultiFab jz(S, amrex::make_alias, Comps[which_slice]["jz"], 1);
        amrex::MultiFab rho(S, amrex::make_alias, Comps[which_slice]["rho"], 1);
        amrex::MultiFab jxx(S, amrex::make_alias, Comps[which_slice]["jxx"], 1);
        amrex::MultiFab jxy(S, amrex::make_alias, Comps[which_slice]["jxy"], 1);
        amrex::MultiFab jyy(S, amrex::make_alias, Comps[which_slice]["jyy"], 1);
        // Extract FabArray for this box
        amrex::FArrayBox& jx_fab = jx[pti];
        amrex::FArrayBox& jy_fab = jy[pti];
        amrex::FArrayBox& jz_fab = jz[pti];
        amrex::FArrayBox& rho_fab = rho[pti];
        amrex::FArrayBox& jxx_fab = jxx[pti];
        amrex::FArrayBox& jxy_fab = jxy[pti];
        amrex::FArrayBox& jyy_fab = jyy[pti];

        // For now: fix the value of the charge
        amrex::Real q =(which_slice == WhichSlice::RhoIons ) ? -plasma.m_charge : plasma.m_charge;

        if        (Hipace::m_depos_order_xy == 0){
                doDepositionShapeN<0, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                          jxx_fab, jxy_fab, jyy_fab,
                                          dx, xyzmin, lo, q, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho,
                                          deposit_j_squared, max_qsa_weighting_factor);
        } else if (Hipace::m_depos_order_xy == 1){
                doDepositionShapeN<1, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                          jxx_fab, jxy_fab, jyy_fab,
                                          dx, xyzmin, lo, q, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho,
                                          deposit_j_squared, max_qsa_weighting_factor);
        } else if (Hipace::m_depos_order_xy == 2){
                doDepositionShapeN<2, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                          jxx_fab, jxy_fab, jyy_fab,
                                          dx, xyzmin, lo, q, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho,
                                          deposit_j_squared, max_qsa_weighting_factor);
        } else if (Hipace::m_depos_order_xy == 3){
                doDepositionShapeN<3, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                          jxx_fab, jxy_fab, jyy_fab,
                                          dx, xyzmin, lo, q, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho,
                                          deposit_j_squared, max_qsa_weighting_factor);
        } else {
            amrex::Abort("unknow deposition order");
        }
    }
}
