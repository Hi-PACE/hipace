#ifndef BEAMDEPOSITCURRENT_H_
#define BEAMDEPOSITCURRENT_H_

#include "particles/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "Constant.H"

void
DepositCurrent (BeamParticleContainer& beam, Fields & fields,
                amrex::Geometry const& gm, int const lev=0);


#endif //  BEAMDEPOSITCURRENT_H_
