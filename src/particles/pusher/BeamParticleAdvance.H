#ifndef BEAMPARTICLEADVANCE_H_
#define BEAMPARTICLEADVANCE_H_

#include "particles/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "Hipace.H"

void
AdvanceBeamParticles (BeamParticleContainer& beam, Fields& fields,
                      amrex::Geometry const& gm, int const lev, const int islice);

#endif //  BEAMPARTICLEADVANCE_H_
