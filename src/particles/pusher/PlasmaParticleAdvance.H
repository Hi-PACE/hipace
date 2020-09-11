#ifndef PLASMAPARTICLEADVANCE_H_
#define PLASMAPARTICLEADVANCE_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "Hipace.H"

/** \brief Gather field values for particles, update the force terms, and push the particles
 * \param[in,out] plasma plasma species to push
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] which_slice defines if this or the next slice is handled
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) will be used
 * \param[in] do_push boolean to define if plasma particles are pushed
 * \param[in] do_update boolean to define if the force terms are updated
 * \param[in] do_shift boolean to define if the force terms are shifted
 * \param[in] lev MR level
 */
void
AdvancePlasmaParticles (PlasmaParticleContainer& plasma, Fields & fields,
                        amrex::Geometry const& gm, const WhichSlice which_slice,
                        const bool temp_slice, const bool do_push, const bool do_update,
                        const bool do_shift, int const lev=0);

/** \brief Resets the particle position x, y, to x_prev, y_prev
 * \param[in,out] plasma plasma species to reset
 * \param[in] lev MR level
 */
void
ResetPlasmaParticles (PlasmaParticleContainer& plasma, int const lev=0, const bool initial=false);

#endif //  PLASMAPARTICLEADVANCE_H_
