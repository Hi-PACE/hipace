#include "PlasmaDepositCurrent.H"

#include "particles/PlasmaParticleContainer.H"
#include "particles/deposition/PlasmaDepositCurrentInner.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"

void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields,
                amrex::Geometry const& gm, int const lev)
{
    BL_PROFILE("DepositCurrent_PlasmaParticleContainer()");
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    PhysConst phys_const = get_phys_const();

    amrex::MultiFab& S = fields.getSlices(lev, 1);

    // Loop over particle boxes
    for ( amrex::MFIter mfi(S); mfi.isValid(); ++mfi )
    {
        // Extract properties associated with the extent of the current box
        // Grow to capture the extent of the particle shape
        amrex::Box tilebox = mfi.tilebox().grow(
            {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields currents
        amrex::MultiFab jx(S, amrex::make_alias, FieldComps::jx, 1);
        amrex::MultiFab jy(S, amrex::make_alias, FieldComps::jy, 1);
        amrex::MultiFab jz(S, amrex::make_alias, FieldComps::jz, 1);
        // Extract FabArray for this box
        amrex::FArrayBox& jx_fab = jx[mfi];
        amrex::FArrayBox& jy_fab = jy[mfi];
        amrex::FArrayBox& jz_fab = jz[mfi];

        // For now: fix the value of the charge
        amrex::Real q = - phys_const.q_e;

        // Call deposition function in each box
        if        (Hipace::m_depos_order_xy == 0){
            doDepositionShapeN<0, 0>( plasma, mfi, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q,
                                          CurrentDepoType::DepositThisSlice );
        } else if (Hipace::m_depos_order_xy == 1){
                doDepositionShapeN<1, 0>( plasma, mfi, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q,
                                          CurrentDepoType::DepositThisSlice );
        } else if (Hipace::m_depos_order_xy == 2){
                doDepositionShapeN<2, 0>( plasma, mfi, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q,
                                          CurrentDepoType::DepositThisSlice );
        } else if (Hipace::m_depos_order_xy == 3){
                doDepositionShapeN<3, 0>( plasma, mfi, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q,
                                          CurrentDepoType::DepositThisSlice );
        } else {
            amrex::Abort("unknow deposition order");
        }
    }
}
