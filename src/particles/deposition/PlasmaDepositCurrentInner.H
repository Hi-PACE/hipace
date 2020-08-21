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
 * \param[in] current_depo_type type of deposition: DepositThisSlice or DepositNextSlice
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
                         const CurrentDepoType current_depo_type)
{
    using namespace amrex::literals;

    const PhysConst phys_const = get_phys_const();

    // Extract particle properties
    const auto& aos = pti.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin();
    const auto& soa = pti.GetStructOfArrays(); // For momenta and weights

    const amrex::Real * const wp = (current_depo_type == CurrentDepoType::DepositThisSlice) ?
        soa.GetRealData(PlasmaIdx::w).data() : soa.GetRealData(PlasmaIdx::w_temp).data();
    const amrex::Real * const uxp = (current_depo_type == CurrentDepoType::DepositThisSlice) ?
        soa.GetRealData(PlasmaIdx::ux).data() : soa.GetRealData(PlasmaIdx::ux_temp).data();
    const amrex::Real * const uyp = (current_depo_type == CurrentDepoType::DepositThisSlice) ?
        soa.GetRealData(PlasmaIdx::uy).data() : soa.GetRealData(PlasmaIdx::uy_temp).data();
    const amrex::Real * const psip = (current_depo_type == CurrentDepoType::DepositThisSlice) ?
        soa.GetRealData(PlasmaIdx::psi).data() : soa.GetRealData(PlasmaIdx::psi_temp).data();
    const amrex::Real * const x_temp = (current_depo_type == CurrentDepoType::DepositThisSlice) ?
        nullptr : soa.GetRealData(PlasmaIdx::x_temp).data();
    const amrex::Real * const y_temp = (current_depo_type == CurrentDepoType::DepositThisSlice) ?
        nullptr : soa.GetRealData(PlasmaIdx::y_temp).data();

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

    // Loop over particles and deposit into jx_fab, jy_fab, jz_fab, and rho_fab
    amrex::ParallelFor(
        pti.numParticles(),
        [=] AMREX_GPU_DEVICE (long ip) {

           // calculate 1/gamma for plasma particles
           const amrex::Real gaminv = (2.0_rt * psip[ip] ) /(1.0_rt
                                                         + uxp[ip]*uxp[ip]*clightsq
                                                         + uyp[ip]*uyp[ip]*clightsq
                                                         + psip[ip]*psip[ip]);

           // calculate plasma particle velocities
           const amrex::Real vx = uxp[ip]*gaminv;
           const amrex::Real vy = uyp[ip]*gaminv;
           const amrex::Real vz = (1._rt - psip[ip]*gaminv) * phys_const.c;

           // calculate charge of the plasma particles
           const amrex::Real wq = q*wp[ip]/(gaminv * psip[ip])*invvol;

           // wqx, wqy wqz are particle current in each direction
           const amrex::Real wqx = wq*vx;
           const amrex::Real wqy = wq*vy;
           const amrex::Real wqz = wq*vz;

            // --- Compute shape factors
            // x direction
            // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            const amrex::Real xmid = (current_depo_type == CurrentDepoType::DepositThisSlice) ?
                (pos_structs[ip].pos(0) - xmin)*dxi : (x_temp[ip] - xmin)*dxi;
            amrex::Real sx_cell[depos_order_xy + 1];
            const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, xmid - 0.5_rt);

            // y direction
            const amrex::Real ymid = (current_depo_type == CurrentDepoType::DepositThisSlice) ?
                (pos_structs[ip].pos(1) - ymin)*dyi : (y_temp[ip] - ymin)*dyi;
            amrex::Real sy_cell[depos_order_xy + 1];
            const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, ymid - 0.5_rt);

            // Deposit current into jx_arr, jy_arr and jz_arr
            for (int iy=0; iy<=depos_order_xy; iy++){
                for (int ix=0; ix<=depos_order_xy; ix++){
                    amrex::Gpu::Atomic::Add(
                        &jx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                        sx_cell[ix]*sy_cell[iy]*wqx);
                    amrex::Gpu::Atomic::Add(
                        &jy_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                        sx_cell[ix]*sy_cell[iy]*wqy);
                    if (current_depo_type == CurrentDepoType::DepositThisSlice)
                    {
                        amrex::Gpu::Atomic::Add(
                            &jz_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                            sx_cell[ix]*sy_cell[iy]*wqz);
                        amrex::Gpu::Atomic::Add(
                            &rho_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                            sx_cell[ix]*sy_cell[iy]*wq);
                    }
                }
            }
        }
        );
}

#endif // PLASMADEPOSITCURRENTINNER_H_
