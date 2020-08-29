#include "PlasmaDepositCurrent.H"

#include "particles/PlasmaParticleContainer.H"
#include "particles/deposition/PlasmaDepositCurrentInner.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"

void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields,
                const CurrentDepoType current_depo_type,
                amrex::Geometry const& gm, int const lev)
{
    BL_PROFILE("DepositCurrent_PlasmaParticleContainer()");
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    PhysConst phys_const = get_phys_const();

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        amrex::Box tilebox = pti.tilebox().grow(2); // Grow to capture the extent of the particle shape

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields currents
        amrex::MultiFab& S = fields.getSlices(lev, 1);
        amrex::MultiFab& next_slice = fields.getSlices(lev, 0);
        amrex::MultiFab jx(S, amrex::make_alias, FieldComps::jx, 1);
        amrex::MultiFab jy(S, amrex::make_alias, FieldComps::jy, 1);
        amrex::MultiFab jx_next(next_slice, amrex::make_alias, FieldComps::jx, 1);
        amrex::MultiFab jy_next(next_slice, amrex::make_alias, FieldComps::jy, 1);
        amrex::MultiFab jz(S, amrex::make_alias, FieldComps::jz, 1);
        amrex::MultiFab rho(S, amrex::make_alias, FieldComps::rho, 1);
        // Extract FabArray for this box
        amrex::FArrayBox& jx_fab = jx[pti];
        amrex::FArrayBox& jy_fab = jy[pti];
        amrex::FArrayBox& jx_next_fab = jx_next[pti];
        amrex::FArrayBox& jy_next_fab = jy_next[pti];
        amrex::FArrayBox& jz_fab = jz[pti];
        amrex::FArrayBox& rho_fab = rho[pti];

        // For now: fix the value of the charge
        amrex::Real q = - phys_const.q_e;

        rho.plus(phys_const.q_e * plasma.m_density, 0, 1);

        // Call deposition function in each box
        if (current_depo_type == CurrentDepoType::DepositThisSlice)
        {
            if        (Hipace::m_depos_order_xy == 0){
                    doDepositionShapeN<0, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin,
                                              lo, q, CurrentDepoType::DepositThisSlice );
            } else if (Hipace::m_depos_order_xy == 1){
                    doDepositionShapeN<1, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin,
                                              lo, q, CurrentDepoType::DepositThisSlice );
            } else if (Hipace::m_depos_order_xy == 2){
                    doDepositionShapeN<2, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin,
                                              lo, q, CurrentDepoType::DepositThisSlice );
            } else if (Hipace::m_depos_order_xy == 3){
                    doDepositionShapeN<3, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin,
                                              lo, q, CurrentDepoType::DepositThisSlice );
            } else {
                amrex::Abort("unknow deposition order");
            }
        }
        else /* (current_depo_type == CurrentDepoType::DepositNextSlice)*/
        {
            if        (Hipace::m_depos_order_xy == 0){
                    doDepositionShapeN<0, 0>( pti, jx_next_fab, jy_next_fab, jz_fab,
                                              rho_fab, dx, xyzmin, lo, q,
                                              CurrentDepoType::DepositNextSlice );
            } else if (Hipace::m_depos_order_xy == 1){
                    doDepositionShapeN<1, 0>( pti, jx_next_fab, jy_next_fab, jz_fab,
                                              rho_fab, dx, xyzmin, lo, q,
                                              CurrentDepoType::DepositNextSlice );
            } else if (Hipace::m_depos_order_xy == 2){
                    doDepositionShapeN<2, 0>( pti, jx_next_fab, jy_next_fab, jz_fab,
                                              rho_fab, dx, xyzmin, lo, q,
                                              CurrentDepoType::DepositNextSlice );
            } else if (Hipace::m_depos_order_xy == 3){
                    doDepositionShapeN<3, 0>( pti, jx_next_fab, jy_next_fab, jz_fab,
                                              rho_fab, dx, xyzmin, lo, q,
                                              CurrentDepoType::DepositNextSlice );
            } else {
                amrex::Abort("unknow deposition order");
            }
                // std::cout << " right in current depo norm jx " << jx_next.norm0() << " norm jy " << jy_next.norm0() << "\n";
        }

    }
}
