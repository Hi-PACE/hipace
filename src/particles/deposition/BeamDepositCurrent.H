#ifndef DEPOSITCURRENT_H_
#define DEPOSITCURRENT_H_

#include <AMReX_Particles.H>
#include "fields/Fields.H"
#include "Constants.H"

template <int nattribs>
void
DepositCurrent (amrex::ParticleContainer<0, 0, nattribs, 0>& species, Fields & fields,
                amrex::Geometry const& gm, int const lev=0);

#endif // DEPOSITCURRENT_H_
