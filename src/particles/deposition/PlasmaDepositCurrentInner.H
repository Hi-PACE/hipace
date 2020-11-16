/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang, Michael Rowan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_PLASMADEPOSITCURRENTINNER_H_
#define HIPACE_PLASMADEPOSITCURRENTINNER_H_

#include "particles/ShapeFactors.H"
#include "Constants.H"
#include "Hipace.H"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

/** \brief Loop over plasma particles in iterator (=box) pti and deposit their current
 * into jx_fab, jy_fab and jz_fab and their density to rho_fab
 *
 * current_depo_type affects the general behavior of this function:
 * - If current_depo_type == WhichSlice::This, deposit jx, jy, jz and rho
 *   with current values of transverse position, wp, uxp, uyp and psip.
 * - If current_depo_type == WhichSlice::Next, deposit jx  and jy only,
 *   with projected values of transverse position, wp, uxp, uyp and psip stored in temp arrays.
 *
 *
 * \tparam depos_order_xy Order of the transverse shape factor for the deposition
 * \tparam depos_order_z Order of the longitudinal shape factor for the deposition
 * \param[in] pti particle iterator, contains data of all particles in a box
 * \param[in,out] jx_fab array of current density jx, on the box corresponding to pti
 * \param[in,out] jy_fab array of current density jy, on the box corresponding to pti
 * \param[in,out] jz_fab array of current density jz, on the box corresponding to pti
 * \param[in,out] rho_fab array of charge density rho, on the box corresponding to pti
 * \param[in] dx cell size in each dimension
 * \param[in] xyzmin lower corner of the box, in physical space
 * \param[in] lo lower corner of the box, in index space
 * \param[in] q elementary charge
 * \param[in] temp_slice if true, the temporary data (x_temp, ...) is used
 * \param[in] deposit_jx_jy if true, deposit to jx and jy
 * \param[in] deposit_jz if true, deposit to jz
 * \param[in] deposit_rho if true, deposit to rho
 * \param[in] max_qsa_weighting_factor maximum allowed weighting factor gamma/(Psi+1)
 */
template <int depos_order_xy, int depos_order_z>
void doDepositionShapeN (const PlasmaParticleIterator& pti,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         amrex::FArrayBox& rho_fab,
                         amrex::Real const * const AMREX_RESTRICT dx,
                         amrex::Real const * const AMREX_RESTRICT xyzmin,
                         amrex::Dim3 const lo,
                         amrex::Real const q,
                         const bool temp_slice,
                         const bool deposit_jx_jy, const bool deposit_jz, const bool deposit_rho,
                         const amrex::Real max_qsa_weighting_factor)
{
    using namespace amrex::literals;

    const PhysConst phys_const = get_phys_const();

    // Extract particle properties
    auto& aos = pti.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin();
    auto& soa = pti.GetStructOfArrays(); // For momenta and weights

    amrex::Real * const wp = soa.GetRealData(PlasmaIdx::w).data() ;
    const amrex::Real * const uxp = (!temp_slice) ?
        soa.GetRealData(PlasmaIdx::ux).data() : soa.GetRealData(PlasmaIdx::ux_temp).data();
    const amrex::Real * const uyp = (!temp_slice) ?
        soa.GetRealData(PlasmaIdx::uy).data() : soa.GetRealData(PlasmaIdx::uy_temp).data();
    const amrex::Real * const psip = (!temp_slice) ?
        soa.GetRealData(PlasmaIdx::psi).data() : soa.GetRealData(PlasmaIdx::psi_temp).data();

    // Extract box properties
    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    const amrex::Real invvol = Hipace::m_normalized_units ? 1. : dxi*dyi*dzi;
    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];

    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::Array4<amrex::Real> const& rho_arr = rho_fab.array();

    constexpr int CELL = amrex::IndexType::CELL;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        jx_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jy_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jz_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        rho_fab.box().type() == amrex::IntVect(CELL, CELL, CELL),
        "jx, jy, jz, and rho must be nodal in all directions."
        );

    int n_qsa_violation = 0;
    amrex::Gpu::DeviceScalar<int> gpu_n_qsa_violation(n_qsa_violation);
    int* p_n_qsa_violation = gpu_n_qsa_violation.dataPtr();

    // Loop over particles and deposit into jx_fab, jy_fab, jz_fab, and rho_fab
    amrex::ParallelFor(
        pti.numParticles(),
        [=] AMREX_GPU_DEVICE (long ip) {

            const amrex::Real psi = psip[ip] *
                phys_const.q_e / (phys_const.m_e * phys_const.c * phys_const.c);

            // calculate 1/gamma for plasma particles
            const amrex::Real gaminv = (2.0_rt * (psi+1.0_rt) ) /(1.0_rt
                                                                  + uxp[ip]*uxp[ip]*clightsq
                                                                  + uyp[ip]*uyp[ip]*clightsq
                                                                  + (psi+1.0_rt)*(psi+1.0_rt));

            if (( 1.0_rt/(gaminv*(psi+1.0_rt)) < 0.0_rt) ||
                 ( 1.0_rt/(gaminv*(psi+1.0_rt)) > max_qsa_weighting_factor))
            {
                 // This particle violates the QSA, discard it and do not deposit its current
                 amrex::Gpu::Atomic::Add(p_n_qsa_violation, 1);
                 wp[ip] = 0.0_rt;
                 return;
            }
            // calculate plasma particle velocities
            const amrex::Real vx = uxp[ip]*gaminv;
            const amrex::Real vy = uyp[ip]*gaminv;
            const amrex::Real vz = - psi * phys_const.c * gaminv;

            // calculate charge of the plasma particles
            const amrex::Real wq = q*wp[ip]/(gaminv * (psi + 1.0_rt))*invvol;

            // wqx, wqy wqz are particle current in each direction
            const amrex::Real wqx = wq*vx;
            const amrex::Real wqy = wq*vy;
            const amrex::Real wqz = wq*vz;

            // --- Compute shape factors
            // x direction
            // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            const amrex::Real xmid = (pos_structs[ip].pos(0) - xmin)*dxi;
            amrex::Real sx_cell[depos_order_xy + 1];
            const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, xmid - 0.5_rt);

            // y direction
            const amrex::Real ymid = (pos_structs[ip].pos(1) - ymin)*dyi;
            amrex::Real sy_cell[depos_order_xy + 1];
            const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, ymid - 0.5_rt);

            // Deposit current into jx_arr, jy_arr and jz_arr
            for (int iy=0; iy<=depos_order_xy; iy++){
                for (int ix=0; ix<=depos_order_xy; ix++){
                    if (deposit_jx_jy) {
                        amrex::Gpu::Atomic::Add(
                            &jx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                            sx_cell[ix]*sy_cell[iy]*wqx);
                        amrex::Gpu::Atomic::Add(
                            &jy_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                            sx_cell[ix]*sy_cell[iy]*wqy);
                    }
                    if (deposit_jz) {
                        amrex::Gpu::Atomic::Add(
                            &jz_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                            sx_cell[ix]*sy_cell[iy]*wqz);
                    }
                    if (deposit_rho) {
                        amrex::Gpu::Atomic::Add(
                            &rho_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                            sx_cell[ix]*sy_cell[iy]*wq);
                    }
                }
            }
            return;
        }
        );
        n_qsa_violation = gpu_n_qsa_violation.dataValue();
        if (n_qsa_violation > 0 && (Hipace::m_verbose >= 4))
            amrex::Print()<< "number of QSA violating particles on this slice: " \
            << n_qsa_violation << "\n";
}

#endif // PLASMADEPOSITCURRENTINNER_H_
