#include "particles/BeamParticleContainer.H"
#include "fields/Fields.H"

void
DepositCurrent (BeamParticleContainer& beam, Fields & fields,
                amrex::Geometry const& gm, int const lev=0)
{

    // Extract properties associated with physical size of the box
    amrex::Real const * dx = gm.CellSize();
    amrex::Real const * xyzmin = gm.ProbLo();

    // Loop over particle boxes
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        amrex::Box tilebox = pti.tilebox().grow(2); // Grow to capture the extent of the particle shape
        const amrex::Dim3 lo = amrex::lbound(tilebox);

        // Extract the fields currents
        amrex::MultiFab& F = fields.getF()[lev];
        amrex::MultiFab jx(F, amrex::make_alias, FieldComps::jx, F.nGrow());
        amrex::MultiFab jy(F, amrex::make_alias, FieldComps::jy, F.nGrow());
        amrex::MultiFab jz(F, amrex::make_alias, FieldComps::jz, F.nGrow());
        // Extract FabArray for this box
        amrex::FArrayBox& jx_fab = jx[pti];
        amrex::FArrayBox& jy_fab = jy[pti];
        amrex::FArrayBox& jz_fab = jz[pti];

        // Call deposition function in each box
        //doChargeDepositionShapeN<2>( pti, jx_fab, jy_fab, jz_fab, dx, xyzmin, lo );
    }

}
