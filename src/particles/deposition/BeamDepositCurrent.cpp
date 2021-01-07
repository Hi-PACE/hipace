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
                     amrex::Geometry const& gm, int const lev, const int islice,
                     amrex::DenseBins<BeamParticleContainer::ParticleType>& bins)
{
    HIPACE_PROFILE("DepositCurrentSlice_BeamParticleContainer()");
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Hipace::m_depos_order_z == 0,
        "Only order 0 deposition is allowed for beam per-slice deposition");

    PhysConst const phys_const = get_phys_const();

    // Loop over particle boxes, transversally. MUST be exactly 1.
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {
        // Assumes '2' == 'z' == 'the long dimension'.
        int islice_local = islice - pti.tilebox().smallEnd(2);

        // Extract properties associated with the extent of the current box
        const amrex::Box tilebox = pti.tilebox().grow(
            {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy,
             Hipace::m_depos_order_z});

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields currents
        amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
        amrex::MultiFab jx(S, amrex::make_alias, FieldComps::jx, 1);
        amrex::MultiFab jy(S, amrex::make_alias, FieldComps::jy, 1);
        amrex::MultiFab jz(S, amrex::make_alias, FieldComps::jz, 1);
        amrex::MultiFab rho(S, amrex::make_alias, FieldComps::rho, 1);

        // Extract FabArray for this box (because there is currently no transverse
        // parallelization, the index we want in the slice multifab is always 0.
        // Fix later.
        amrex::FArrayBox& jx_fab = jx[0];
        amrex::FArrayBox& jy_fab = jy[0];
        amrex::FArrayBox& jz_fab = jz[0];
        amrex::FArrayBox& rho_fab = rho[0];

        // For now: fix the value of the charge
        const amrex::Real q = - phys_const.q_e;

        // Call deposition function in each box
        if        (Hipace::m_depos_order_xy == 0){
            doDepositionShapeN<0, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                      dx, xyzmin, lo, q, islice_local, bins);
        } else if (Hipace::m_depos_order_xy == 1){
            doDepositionShapeN<1, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                      dx, xyzmin, lo, q, islice_local, bins);
        } else if (Hipace::m_depos_order_xy == 2){
            doDepositionShapeN<2, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                      dx, xyzmin, lo, q, islice_local, bins);
        } else if (Hipace::m_depos_order_xy == 3){
            doDepositionShapeN<3, 0>( pti, jx_fab, jy_fab, jz_fab, rho_fab,
                                      dx, xyzmin, lo, q, islice_local, bins);
        } else {
            amrex::Abort("unknown deposition order");
        }
    }
}
