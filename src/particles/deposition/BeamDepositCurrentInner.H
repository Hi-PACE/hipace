#ifndef HIPACE_CURRENTDEPOSITION_H_
#define HIPACE_CURRENTDEPOSITION_H_

#include "particles/ShapeFactors.H"
#include "Constants.H"
#include "Hipace.H"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

template <int depos_order_xy, int depos_order_z>
void doDepositionShapeN(const BeamParticleIterator& pti,
                        amrex::FArrayBox& jx_fab,
                        amrex::FArrayBox& jy_fab,
                        amrex::FArrayBox& jz_fab,
                        amrex::Real const * AMREX_RESTRICT dx,
                        amrex::Real const * AMREX_RESTRICT xyzmin,
                        amrex::Dim3 const lo,
                        amrex::Real const q)
{
    using namespace amrex::literals;

    PhysConst phys_const = get_phys_const();

    // Extract particle properties
    const auto& aos = pti.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin();
    const auto& soa = pti.GetStructOfArrays(); // For momenta and weights
    const auto  wp = soa.GetRealData(BeamIdx::w).data();
    const auto uxp = soa.GetRealData(BeamIdx::ux).data();
    const auto uyp = soa.GetRealData(BeamIdx::uy).data();
    const auto uzp = soa.GetRealData(BeamIdx::uz).data();

    // Extract box properties
    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    const amrex::Real invvol = Hipace::m_normalized_units? 1. : dxi*dyi*dzi;
    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];
    const amrex::Real zmin = xyzmin[2];

    const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();

    constexpr int CELL = amrex::IndexType::CELL;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        jx_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jy_fab.box().type() == amrex::IntVect(CELL, CELL, CELL) &&
        jz_fab.box().type() == amrex::IntVect(CELL, CELL, CELL),
        "jx, jy and jz must be nodal in all directions."
        );

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
        pti.numParticles(),
        [=] AMREX_GPU_DEVICE (long ip) {
            // --- Get particle quantities
            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                         + uyp[ip]*uyp[ip]*clightsq
                                                         + uzp[ip]*uzp[ip]*clightsq);
            const amrex::Real wq = q*wp[ip];

            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;
            // wqx, wqy wqz are particle current in each direction
            const amrex::Real wqx = wq*invvol*vx;
            const amrex::Real wqy = wq*invvol*vy;
            const amrex::Real wqz = wq*invvol*vz;

            // --- Compute shape factors
            // x direction
            const amrex::Real xmid = (pos_structs[ip].pos(0) - xmin)*dxi;
            // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            amrex::Real sx_cell[depos_order_xy + 1];
            const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, xmid - 0.5_rt);
            
            // y direction
            const amrex::Real ymid = (pos_structs[ip].pos(1) - ymin)*dyi;
            amrex::Real sy_cell[depos_order_xy + 1];
            const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, ymid - 0.5_rt);

            // z direction
            const amrex::Real zmid = (pos_structs[ip].pos(2) - zmin)*dzi;
            amrex::Real sz_cell[depos_order_z + 1];
            const int l_cell = compute_shape_factor<depos_order_z>(sz_cell, zmid - 0.5_rt);

            // Deposit current into jx_arr, jy_arr and jz_arr
            for (int iz=0; iz<=depos_order_z; iz++){
                for (int iy=0; iy<=depos_order_xy; iy++){
                    for (int ix=0; ix<=depos_order_xy; ix++){
                        amrex::Gpu::Atomic::Add(
                            &jx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz),
                            sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqx);
                        amrex::Gpu::Atomic::Add(
                            &jy_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz),
                            sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqy);
                        amrex::Gpu::Atomic::Add(
                            &jz_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, lo.z+l_cell+iz),
                            sx_cell[ix]*sy_cell[iy]*sz_cell[iz]*wqz);
                    }
                }
            }
        }
        );
}

#endif // CURRENTDEPOSITION_H_
