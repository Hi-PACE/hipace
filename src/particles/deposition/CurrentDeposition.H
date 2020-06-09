#ifndef HIPACE_CURRENTDEPOSITION_H_
#define HIPACE_CURRENTDEPOSITION_H_

#include "particles/ShapeFactors.H"
#include "Constants.H"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

template <int depos_order>
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

    // Extract particle properties
    auto& aos = pti.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin();
    auto& soa = pti.GetStructOfArrays(); // For momenta and weights
    auto  wp = soa.GetRealData(BeamIdx::w).data();
    auto uxp = soa.GetRealData(BeamIdx::ux).data();
    auto uyp = soa.GetRealData(BeamIdx::uy).data();
    auto uzp = soa.GetRealData(BeamIdx::uz).data();

    // Extract box properties
    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    const amrex::Real invvol = dxi*dyi*dzi;
    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];
    const amrex::Real zmin = xyzmin[2];

    const amrex::Real clightsq = 1.0_rt/(PhysConst::c*PhysConst::c);

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    constexpr int zdir = (AMREX_SPACEDIM - 1);
    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
        pti.numParticles(),
        [=] AMREX_GPU_DEVICE (long ip) {
            // --- Get particle quantities
            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                         + uyp[ip]*uyp[ip]*clightsq
                                                         + uzp[ip]*uzp[ip]*clightsq);
            amrex::Real wq  = q*wp[ip];

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
            // j_j[xyz] leftmost grid point in x that the particle touches for the centering of each current
            // sx_j[xyz] shape factor along x for the centering of each current
            // There are only two possible centerings, node or cell centered, so at most only two shape factor
            // arrays will be needed.
            amrex::Real sx_node[depos_order + 1];
            amrex::Real sx_cell[depos_order + 1];
            int j_node;
            int j_cell;
            if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE) {
                j_node = compute_shape_factor<depos_order>(sx_node, xmid);
            }
            if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL) {
                j_cell = compute_shape_factor<depos_order>(sx_cell, xmid - 0.5_rt);
            }
            const amrex::Real (&sx_jx)[depos_order + 1] = ((jx_type[0] == NODE) ? sx_node : sx_cell);
            const amrex::Real (&sx_jy)[depos_order + 1] = ((jy_type[0] == NODE) ? sx_node : sx_cell);
            const amrex::Real (&sx_jz)[depos_order + 1] = ((jz_type[0] == NODE) ? sx_node : sx_cell);
            int const j_jx = ((jx_type[0] == NODE) ? j_node : j_cell);
            int const j_jy = ((jy_type[0] == NODE) ? j_node : j_cell);
            int const j_jz = ((jz_type[0] == NODE) ? j_node : j_cell);

            // y direction
            const amrex::Real ymid = (pos_structs[ip].pos(1) - ymin)*dyi;
            amrex::Real sy_node[depos_order + 1];
            amrex::Real sy_cell[depos_order + 1];
            int k_node;
            int k_cell;
            if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE) {
                k_node = compute_shape_factor<depos_order>(sy_node, ymid);
            }
            if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL) {
                k_cell = compute_shape_factor<depos_order>(sy_cell, ymid - 0.5);
            }
            const amrex::Real (&sy_jx)[depos_order + 1] = ((jx_type[1] == NODE) ? sy_node : sy_cell);
            const amrex::Real (&sy_jy)[depos_order + 1] = ((jy_type[1] == NODE) ? sy_node : sy_cell);
            const amrex::Real (&sy_jz)[depos_order + 1] = ((jz_type[1] == NODE) ? sy_node : sy_cell);
            int const k_jx = ((jx_type[1] == NODE) ? k_node : k_cell);
            int const k_jy = ((jy_type[1] == NODE) ? k_node : k_cell);
            int const k_jz = ((jz_type[1] == NODE) ? k_node : k_cell);

            // z direction
            const amrex::Real zmid = (pos_structs[ip].pos(2) - zmin)*dzi;
            amrex::Real sz_node[depos_order + 1];
            amrex::Real sz_cell[depos_order + 1];
            int l_node;
            int l_cell;
            if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE) {
                l_node = compute_shape_factor<depos_order>(sz_node, zmid);
            }
            if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL) {
                l_cell = compute_shape_factor<depos_order>(sz_cell, zmid - 0.5);
            }
            const amrex::Real (&sz_jx)[depos_order + 1] = ((jx_type[zdir] == NODE) ? sz_node : sz_cell);
            const amrex::Real (&sz_jy)[depos_order + 1] = ((jy_type[zdir] == NODE) ? sz_node : sz_cell);
            const amrex::Real (&sz_jz)[depos_order + 1] = ((jz_type[zdir] == NODE) ? sz_node : sz_cell);
            int const l_jx = ((jx_type[zdir] == NODE) ? l_node : l_cell);
            int const l_jy = ((jy_type[zdir] == NODE) ? l_node : l_cell);
            int const l_jz = ((jz_type[zdir] == NODE) ? l_node : l_cell);

            // Deposit current into jx_arr, jy_arr and jz_arr
            for (int iz=0; iz<=depos_order; iz++){
                for (int iy=0; iy<=depos_order; iy++){
                    for (int ix=0; ix<=depos_order; ix++){
                        amrex::Gpu::Atomic::Add(
                            &jx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz),
                            sx_jx[ix]*sy_jx[iy]*sz_jx[iz]*wqx);
                        amrex::Gpu::Atomic::Add(
                            &jy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz),
                            sx_jy[ix]*sy_jy[iy]*sz_jy[iz]*wqy);
                        amrex::Gpu::Atomic::Add(
                            &jz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz),
                            sx_jz[ix]*sy_jz[iy]*sz_jz[iz]*wqz);
                    }
                }
            }
        }
        );
}

#endif // CURRENTDEPOSITION_H_
