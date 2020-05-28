#include "particles/BeamParticleContainer.H"
#include "fields/Fields.H"

void
DepositCurrent (BeamParticleContainer& beam, Fields& fields, const int lev=0)
{

    // Loop over particle boxes
    for (BeamParticleIterator pti(beam, lev); pti.isValid(); ++pti)
    {
        // Extract corresponding fields for this box
        amrex::FArrayBox& fields_in_box = fields.getF()[lev][pti];

        // Call deposition function in each box
        //doChargeDepositionShapeN<2>( pti, fields_in_box );
    }

}
