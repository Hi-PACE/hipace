#ifndef BEAMDEPOSITCURRENT_H_
#define BEAMDEPOSITCURRENT_H_

#include "particles/BeamParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"

#include <AMReX_DenseBins.H>

/** Depose current of beam particles on a single slice
 * \param[in] beam species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 * \param[in] islice index of the slice on which the beam particles are pushed
 * \param[in] bins beam particle container bins, to push only the beam particles on slice islice
 */
void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Geometry const& gm, int const lev, const int islice,
                     amrex::DenseBins<BeamParticleContainer::ParticleType>& bins);


#endif //  BEAMDEPOSITCURRENT_H_
