#include "BeamDepositCurrent.H"

// #include "particles/BeamParticleContainer.H"
#include "particles/deposition/CurrentDeposition.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"

#include <AMReX_Particles.H>

template <int nattribs>
void
DepositCurrent (amrex::ParticleContainer<0, 0, nattribs, 0>& species, Fields & fields,
                amrex::Geometry const& gm, int const lev)
{

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    // Loop over particle boxes
    for (amrex::ParIter<0, 0, nattribs> pti(species, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        amrex::Box tilebox = pti.tilebox().grow(2); // Grow to capture the extent of the particle shape

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields currents
        amrex::MultiFab& F = fields.getF()[lev];
        amrex::MultiFab jx(F, amrex::make_alias, FieldComps::jx, F.nGrow());
        amrex::MultiFab jy(F, amrex::make_alias, FieldComps::jy, F.nGrow());
        amrex::MultiFab jz(F, amrex::make_alias, FieldComps::jz, F.nGrow());
        // Extract FabArray for this box
        amrex::FArrayBox& jx_fab = jx[pti];
        amrex::FArrayBox& jy_fab = jy[pti];
        amrex::FArrayBox& jz_fab = jz[pti];

        // For now: fix the value of the charge
        amrex::Real q = - PhysConst::q_e;

        // Call deposition function in each box
        if        (Hipace::m_depos_order_xy == 0){
            if        (Hipace::m_depos_order_z == 0){
                doDepositionShapeN<0, 0, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 1){
                doDepositionShapeN<0, 1, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 2){
                doDepositionShapeN<0, 2, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 3){
                doDepositionShapeN<0, 3, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else {
                amrex::Abort("unknow deposition order");
            }
        } else if (Hipace::m_depos_order_xy == 1){
            if        (Hipace::m_depos_order_z == 0){
                doDepositionShapeN<1, 0, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 1){
                doDepositionShapeN<1, 1, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 2){
                doDepositionShapeN<1, 2, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 3){
                doDepositionShapeN<1, 3, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else {
                amrex::Abort("unknow deposition order");
            }
        } else if (Hipace::m_depos_order_xy == 2){
            if        (Hipace::m_depos_order_z == 0){
                doDepositionShapeN<2, 0, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 1){
                doDepositionShapeN<2, 1, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 2){
                doDepositionShapeN<2, 2, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 3){
                doDepositionShapeN<2, 3, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else {
                amrex::Abort("unknow deposition order");
            }
        } else if (Hipace::m_depos_order_xy == 3){
            if        (Hipace::m_depos_order_z == 0){
                doDepositionShapeN<3, 0, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 1){
                doDepositionShapeN<3, 1, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 2){
                doDepositionShapeN<3, 2, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else if (Hipace::m_depos_order_z == 3){
                doDepositionShapeN<3, 3, BeamIdx::nattribs>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo, q, BeamIdx::w, BeamIdx::ux, BeamIdx::uy, BeamIdx::ux);
            } else {
                amrex::Abort("unknow deposition order m_depos_order_z");
            }
        } else {
            amrex::Abort("unknow deposition order m_depos_order_xy");
        }
    }
}
