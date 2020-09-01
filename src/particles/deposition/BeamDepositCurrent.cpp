#include "BeamDepositCurrent.H"

#include "particles/BeamParticleContainer.H"
#include "particles/deposition/BeamDepositCurrentInner.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_DenseBins.H>

void
DepositCurrent (BeamParticleContainer& beam, Fields & fields,
                amrex::Geometry const& gm, int const lev)
{
    BL_PROFILE("DepositCurrent_BeamParticleContainer()");
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    PhysConst const phys_const = get_phys_const();

    // Loop over particle boxes
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        const amrex::Box tilebox = pti.tilebox().grow(
            {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy,
             Hipace::m_depos_order_z});


        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields currents
        amrex::MultiFab& F = fields.getF(lev);
        amrex::MultiFab jx(F, amrex::make_alias, FieldComps::jx, 1);
        amrex::MultiFab jy(F, amrex::make_alias, FieldComps::jy, 1);
        amrex::MultiFab jz(F, amrex::make_alias, FieldComps::jz, 1);
        amrex::MultiFab rho(F, amrex::make_alias, FieldComps::rho, 1);
        // Extract FabArray for this box
        amrex::FArrayBox& jx_fab = jx[pti];
        amrex::FArrayBox& jy_fab = jy[pti];
        amrex::FArrayBox& jz_fab = jz[pti];
        amrex::FArrayBox& rho_fab = rho[pti];

        // For now: fix the value of the charge
        const amrex::Real q = - phys_const.q_e;
        // Declare a DenseBins to pass it to doDepositionShapeN, although it will not be used.
        amrex::DenseBins<BeamParticleContainer::ParticleType> bins;

        // Call deposition function in each box
        if        (Hipace::m_depos_order_xy == 0){
            if        (Hipace::m_depos_order_z == 0){
                doDepositionShapeN<0, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 1){
                doDepositionShapeN<0, 1>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 2){
                doDepositionShapeN<0, 2>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 3){
                doDepositionShapeN<0, 3>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else {
                amrex::Abort("unknow deposition order");
            }
        } else if (Hipace::m_depos_order_xy == 1){
            if        (Hipace::m_depos_order_z == 0){
                doDepositionShapeN<1, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 1){
                doDepositionShapeN<1, 1>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 2){
                doDepositionShapeN<1, 2>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 3){
                doDepositionShapeN<1, 3>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else {
                amrex::Abort("unknow deposition order");
            }
        } else if (Hipace::m_depos_order_xy == 2){
            if        (Hipace::m_depos_order_z == 0){
                doDepositionShapeN<2, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 1){
                doDepositionShapeN<2, 1>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 2){
                doDepositionShapeN<2, 2>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 3){
                doDepositionShapeN<2, 3>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else {
                amrex::Abort("unknow deposition order");
            }
        } else if (Hipace::m_depos_order_xy == 3){
            if        (Hipace::m_depos_order_z == 0){
                doDepositionShapeN<3, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 1){
                doDepositionShapeN<3, 1>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 2){
                doDepositionShapeN<3, 2>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else if (Hipace::m_depos_order_z == 3){
                doDepositionShapeN<3, 3>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, 0, bins, false);
            } else {
                amrex::Abort("unknow deposition order m_depos_order_z");
            }
        } else {
            amrex::Abort("unknow deposition order m_depos_order_xy");
        }
    }
}

void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Geometry const& gm, int const lev, const int islice,
                     amrex::DenseBins<BeamParticleContainer::ParticleType>& bins)
{
    BL_PROFILE("DepositCurrentSlice_BeamParticleContainer()");
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Hipace::m_depos_order_z == 0,
        "Only order 0 deposition is allowed for beam per-slice deposition");

    PhysConst const phys_const = get_phys_const();

    // Loop over particle boxes, transversally. MUST be exactly 1.
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        const amrex::Box tilebox = pti.tilebox().grow(
            {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy,
             Hipace::m_depos_order_z});

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields currents
        amrex::MultiFab& S = fields.getSlices(lev, 1);
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
        const amrex::Real q = - phys_const.q_e;

        // Call deposition function in each box
        if        (Hipace::m_depos_order_xy == 0){
            doDepositionShapeN<0, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, islice, bins, true);
        } else if (Hipace::m_depos_order_xy == 1){
            doDepositionShapeN<1, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, islice, bins, true);
        } else if (Hipace::m_depos_order_xy == 2){
            doDepositionShapeN<2, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, islice, bins, true);
        } else if (Hipace::m_depos_order_xy == 3){
            doDepositionShapeN<3, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab, dx, xyzmin, lo, q, islice, bins, true);
        } else {
            amrex::Abort("unknow deposition order");
        }
    }
}
