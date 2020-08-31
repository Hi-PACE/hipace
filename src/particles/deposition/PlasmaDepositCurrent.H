#ifndef PLASMADEPOSITCURRENT_H_
#define PLASMADEPOSITCURRENT_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"

/** Depose current of particles in species plasma into the current 2D slice in fields
 * \param[in] plasma species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] current_depo_type defines if this or the next slice is handled
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 */
void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields,
                const ToSlice current_depo_type,
                amrex::Geometry const& gm, int const lev=0);


#endif //  PLASMADEPOSITCURRENT_H_
