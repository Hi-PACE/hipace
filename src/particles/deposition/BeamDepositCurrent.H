#ifndef BEAMDEPOSITCURRENT_H_
#define BEAMDEPOSITCURRENT_H_

#include "particles/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "Constants.H"

#include <AMReX_DenseBins.H>

/** Depose current of particles in species beam into the 3D array in fields
 * \param[in] beam species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 */
void
DepositCurrent (BeamParticleContainer& beam, Fields& fields,
                amrex::Geometry const& gm, int const lev);

void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Geometry const& gm, int const lev, const int islice,
                     amrex::DenseBins<BeamParticleContainer::ParticleType>& bins);


#endif //  BEAMDEPOSITCURRENT_H_
