#ifndef PLASMADEPOSITCURRENT_H_
#define PLASMADEPOSITCURRENT_H_

#include "particles/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"

/** Depose current of particles in species plasma into the current 2D slice in fields
 * \param[in] plasma species of which the current is deposited
 * \param[in,out] fields the general field class, modified by this function
 * \param[in] which_slice defines if this or the next slice is handled
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) is used
 * \param[in] deposit_jx_jy if true, deposit to jx and jy
 * \param[in] deposit_jz if true, deposit to jz
 * \param[in] deposit_rho if true, deposit to rho
 * \param[in] gm Geometry of the simulation, to get the cell size etc.
 * \param[in] lev MR level
 */
void
DepositCurrent (PlasmaParticleContainer& plasma, Fields & fields,
                const WhichSlice which_slice, const bool temp_slice,
                const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                amrex::Geometry const& gm, int const lev);


#endif //  PLASMADEPOSITCURRENT_H_
