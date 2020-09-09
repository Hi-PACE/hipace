#ifndef FFT_POISSON_SOLVER_UTIL_H_
#define FFT_POISSON_SOLVER_UTIL_H_

#include <AMReX_MultiFab.H>


/** \brief this function takes an nx*ny multifab src and returns an symmetrized
 * (2nx+2)(2ny+2) array. This prepares the array as an input to a 2D DFT, which
 * yields the same result of the corrensponding 2D DST.
 * \param[in] src input multifab
 * \param[in,out] dst symmetrized output multifab
 * \param[in] geom Geometry
 * \param[in] scomp source component
 * \param[in] dcomp destination component
 */
void
ExpandRealData (const amrex::MultiFab& src, amrex::MultiFab& dst, amrex::Geometry const& geom,
                const int scomp=0, const int dcomp=0);

/** \brief this function takes an (2nx+2)(2ny+2) multifab src and extracts
*  nx*ny array after a 2D DFT. The returned array is the result of a 2D DST.
* \param[in] src input multifab
* \param[in,out] dst symmetrized output multifab
* \param[in] geom Geometry
* \param[in] scomp source component
* \param[in] dcomp destination component
*/
void
ShrinkRealData (const amrex::MultiFab& src, amrex::MultiFab& dst, amrex::Geometry const& geom,
                const int scomp=0, const int dcomp=0);

#endif
