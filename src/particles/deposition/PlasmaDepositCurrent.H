#ifndef PLASMADEPOSITCURRENT_H_
#define PLASMADEPOSITCURRENT_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "Constants.H"

void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields,
                amrex::Geometry const& gm, int const lev=0);


#endif //  PLASMADEPOSITCURRENT_H_
