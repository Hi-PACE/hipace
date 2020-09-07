#include "PlasmaDepositCurrent.H"

#include "particles/PlasmaParticleContainer.H"
#include "particles/deposition/PlasmaDepositCurrentInner.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"

void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields,
                const WhichSlice which_slice, const bool temp_slice,
                const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                amrex::Geometry const& gm, int const lev)
{
    HIPACE_PROFILE("DepositCurrent_PlasmaParticleContainer()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    which_slice == WhichSlice::This || which_slice == WhichSlice::Next ||
    which_slice == WhichSlice::RhoIons,
    "Current deposition can only be done in this slice (WhichSlice::This), the next slice "
    " (WhichSlice::Next) or for the ion charge deposition (WhichSLice::RhoIons)");

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    PhysConst phys_const = get_phys_const();

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
        amrex::MultiFab jx(S, amrex::make_alias, FieldComps::jx, 1);
        amrex::MultiFab jy(S, amrex::make_alias, FieldComps::jy, 1);
        amrex::MultiFab jz(S, amrex::make_alias, FieldComps::jz, 1);
        amrex::MultiFab rho(S, amrex::make_alias, FieldComps::rho, 1);
        // Extract FabArray for this box
        amrex::FArrayBox& jx_fab = jx[pti];
        amrex::FArrayBox& jy_fab = jy[pti];
        amrex::FArrayBox& jz_fab = jz[pti];
        amrex::FArrayBox& rho_fab = rho[pti];

        // For now: fix the value of the charge
        amrex::Real q =(which_slice == WhichSlice::RhoIons ) ? phys_const.q_e : - phys_const.q_e;

        // Call deposition function in each box
        // Deposit ion charge density, assumed uniform
        // if (which_slice == WhichSlice::This)
        // {
        //     // rho.plus(phys_const.q_e * plasma.m_density, 0, 1);
        // }


        if        (Hipace::m_depos_order_xy == 0){
                doDepositionShapeN<0, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                          dx, xyzmin, lo, q, which_slice, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho);
        } else if (Hipace::m_depos_order_xy == 1){
                doDepositionShapeN<1, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                          dx, xyzmin, lo, q, which_slice, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho);
        } else if (Hipace::m_depos_order_xy == 2){
                doDepositionShapeN<2, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                          dx, xyzmin, lo, q, which_slice, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho);
        } else if (Hipace::m_depos_order_xy == 3){
                doDepositionShapeN<3, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                          dx, xyzmin, lo, q, which_slice, temp_slice,
                                          deposit_jx_jy, deposit_jz, deposit_rho);
        } else {
            amrex::Abort("unknow deposition order");
        }
    }
}
