#ifndef PLASMAPARTICLEPUSHER_H_
#define PLASMAPARTICLEPUSHER_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "FieldGather.H"

void
UpdateForcePushParticles (PlasmaParticleContainer& plasma, Fields & fields,
                          amrex::Geometry const& gm, int const lev=0);
// void
// UpdateForceTerms (PlasmaParticleContainer& plasma, Fields & fields,
//                 amrex::Geometry const& gm, int const lev=0);
//
// void
// PlasmaParticlePusher (PlasmaParticleContainer& plasma, //Fields & fields,
//                 amrex::Geometry const& gm, int const lev=0);



#endif //  PLASMAPARTICLEPUSHER_H_
