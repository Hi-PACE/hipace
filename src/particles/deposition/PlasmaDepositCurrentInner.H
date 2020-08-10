#ifndef HIPACE_PLASMADEPOSITCURRENTINNER_H_
#define HIPACE_PLASMADEPOSITCURRENTINNER_H_

#include "particles/ShapeFactors.H"
#include "Constants.H"
#include "Hipace.H"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

template <int depos_order_xy, int depos_order_z>
void doDepositionShapeN (const PlasmaParticleIterator& pti,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         amrex::Real const * AMREX_RESTRICT dx,
                         amrex::Real const * AMREX_RESTRICT xyzmin,
                         amrex::Dim3 const lo,
                         amrex::Real const q,
                         const CurrentDepoType CurrentDepoType)
{
    using namespace amrex::literals;

    PhysConst phys_const = get_phys_const();

    // Extract particle properties
    const auto& aos = pti.GetArrayOfStructs(); // For positions
    const auto& pos_structs = aos.begin();
    const auto& soa = pti.GetStructOfArrays(); // For momenta and weights

    const amrex::Real* wp = (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)? soa.GetRealData(PlasmaIdx::w).data() : soa.GetRealData(PlasmaIdx::w_temp).data();;
    const amrex::Real* uxp = (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)? soa.GetRealData(PlasmaIdx::ux).data() : soa.GetRealData(PlasmaIdx::ux_temp).data();
    const amrex::Real* uyp = (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)? soa.GetRealData(PlasmaIdx::uy).data() : soa.GetRealData(PlasmaIdx::uy_temp).data();
    const amrex::Real* psip = (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)? soa.GetRealData(PlasmaIdx::psi).data() :soa.GetRealData(PlasmaIdx::psi_temp).data();
    const amrex::Real* x_temp = (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)? nullptr :  soa.GetRealData(PlasmaIdx::x_temp).data();
    const amrex::Real* y_temp = (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)? nullptr :  soa.GetRealData(PlasmaIdx::y_temp).data();

    // Extract box properties
    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    const amrex::Real invvol = Hipace::m_normalized_units? 1. : dxi*dyi*dzi;
    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];

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

           // calculate 1/gamma for plasma particles
           const amrex::Real gaminv = (2.0_rt * psip[ip] ) /(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                         + uyp[ip]*uyp[ip]*clightsq
                                                         + psip[ip]*psip[ip]);
           std::cout<<"gamma "<<1./gaminv<<'\n';

           // calculate plasma particle velocities
           const amrex::Real vx  = uxp[ip]*gaminv;
           const amrex::Real vy  = uyp[ip]*gaminv;
           const amrex::Real vz  = (1-psip[ip])*gaminv;
           std::cout<<"vx "<<vx<<'\n';
           std::cout<<"vy "<<vy<<'\n';
           std::cout<<"vz "<<vz<<'\n';
           // calculate charge of the plasma particles
           std::cout<<"gamma / psip[ip] "<<1./gaminv/psip[ip]<<'\n';
           const amrex::Real wq = q*wp[ip]/(gaminv * psip[ip]);
           std::cout<<"wq "<<wq<<'\n';
           std::cout<<"q "<<q<<'\n';
           std::cout<<"wp[ip] "<<wp[ip]<<'\n';
           std::cout<<"psip[ip] "<<psip[ip]<<'\n';
           std::cout<<"invvol "<<invvol<<'\n';
           // wqx, wqy wqz are particle current in each direction
           const amrex::Real wqx = wq*invvol*vx;
           const amrex::Real wqy = wq*invvol*vy;
           const amrex::Real wqz = wq*invvol*vz;
           std::cout<<"wqx "<<wqx<<'\n';
           std::cout<<"wqy "<<wqy<<'\n';
           std::cout<<"wqz "<<wqz<<'\n';

            // --- Compute shape factors
            // x direction
            // j_cell leftmost cell in x that the particle touches. sx_cell shape factor along x
            const amrex::Real xmid = (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)? (pos_structs[ip].pos(0) - xmin)*dxi : (x_temp[ip] - xmin)*dxi;
            amrex::Real sx_cell[depos_order_xy + 1];
            const int j_cell = compute_shape_factor<depos_order_xy>(sx_cell, xmid - 0.5_rt);

            // y direction
            const amrex::Real ymid = (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)? (pos_structs[ip].pos(1) - ymin)*dyi : (y_temp[ip] - ymin)*dyi;
            amrex::Real sy_cell[depos_order_xy + 1];
            const int k_cell = compute_shape_factor<depos_order_xy>(sy_cell, ymid - 0.5_rt);

            // Deposit current into jx_arr, jy_arr and jz_arr
            for (int iy=0; iy<=depos_order_xy; iy++){
                for (int ix=0; ix<=depos_order_xy; ix++){
//                    std::cout<<"lo.z "<<lo.z<<'\n';
                    amrex::Gpu::Atomic::Add(
                        &jx_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                        sx_cell[ix]*sy_cell[iy]*wqx);
                    amrex::Gpu::Atomic::Add(
                        &jy_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                        sx_cell[ix]*sy_cell[iy]*wqy);
                    if (CurrentDepoType == CurrentDepoType::DepositCurrentSlice)
                    {
                      amrex::Gpu::Atomic::Add(
                          &jz_arr(lo.x+j_cell+ix, lo.y+k_cell+iy, 0),
                          sx_cell[ix]*sy_cell[iy]*wqz);
                    }
                }
            }
        }
        );
}

#endif // PLASMADEPOSITCURRENTINNER_H_
