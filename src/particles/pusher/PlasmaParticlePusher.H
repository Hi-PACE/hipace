#ifndef PLASMAPARTICLEPUSHER_H_
#define PLASMAPARTICLEPUSHER_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "FieldGather.H"

/** \brief Gather field values for particles, update the force terms, and push the particles
 * \param[in,out] plasma plasma species to push
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 */
void
UpdateForcePushParticles (PlasmaParticleContainer& plasma, Fields & fields,
                          amrex::Geometry const& gm, int const lev=0);

#endif //  PLASMAPARTICLEPUSHER_H_
