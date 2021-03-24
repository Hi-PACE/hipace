#include "PlasmaParticleAdvance.H"

#include "particles/PlasmaParticleContainer.H"
#include "FieldGather.H"
#include "PushPlasmaParticles.H"
#include "UpdateForceTerms.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "GetAndSetPosition.H"
#include "utils/HipaceProfilerWrapper.H"

void
AdvancePlasmaParticles (PlasmaParticleContainer& plasma, Fields & fields,
                        amrex::Geometry const& gm, const bool temp_slice, const bool do_push,
                        const bool do_update, const bool do_shift, int const lev)
{
    HIPACE_PROFILE("UpdateForcePushParticles_PlasmaParticleContainer()");
    using namespace amrex::literals;

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();
    const PhysConst phys_const = get_phys_const();

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        // Grow to capture the extent of the particle shape
        amrex::Box tilebox = pti.tilebox().grow(
            {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields
        const amrex::MultiFab& S = fields.getSlices(lev, WhichSlice::This);
        const amrex::MultiFab exmby(S, amrex::make_alias, Comps[WhichSlice::This]["ExmBy"], 1);
        const amrex::MultiFab eypbx(S, amrex::make_alias, Comps[WhichSlice::This]["EypBx"], 1);
        const amrex::MultiFab ez(S, amrex::make_alias, Comps[WhichSlice::This]["Ez"], 1);
        const amrex::MultiFab bx(S, amrex::make_alias, Comps[WhichSlice::This]["Bx"], 1);
        const amrex::MultiFab by(S, amrex::make_alias, Comps[WhichSlice::This]["By"], 1);
        const amrex::MultiFab bz(S, amrex::make_alias, Comps[WhichSlice::This]["Bz"], 1);
        // Extract FabArray for this box
        const amrex::FArrayBox& exmby_fab = exmby[pti];
        const amrex::FArrayBox& eypbx_fab = eypbx[pti];
        const amrex::FArrayBox& ez_fab = ez[pti];
        const amrex::FArrayBox& bx_fab = bx[pti];
        const amrex::FArrayBox& by_fab = by[pti];
        const amrex::FArrayBox& bz_fab = bz[pti];
        // Extract field array from FabArray
        amrex::Array4<const amrex::Real> const& exmby_arr = exmby_fab.array();
        amrex::Array4<const amrex::Real> const& eypbx_arr = eypbx_fab.array();
        amrex::Array4<const amrex::Real> const& ez_arr = ez_fab.array();
        amrex::Array4<const amrex::Real> const& bx_arr = bx_fab.array();
        amrex::Array4<const amrex::Real> const& by_arr = by_fab.array();
        amrex::Array4<const amrex::Real> const& bz_arr = bz_fab.array();

        const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
        const amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};

        auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        // loading the data
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();

        amrex::Real * const x_prev = soa.GetRealData(PlasmaIdx::x_prev).data();
        amrex::Real * const y_prev = soa.GetRealData(PlasmaIdx::y_prev).data();
        amrex::Real * const ux_temp = soa.GetRealData(PlasmaIdx::ux_temp).data();
        amrex::Real * const uy_temp = soa.GetRealData(PlasmaIdx::uy_temp).data();
        amrex::Real * const psi_temp = soa.GetRealData(PlasmaIdx::psi_temp).data();

        amrex::Real * const Fx1 = soa.GetRealData(PlasmaIdx::Fx1).data();
        amrex::Real * const Fy1 = soa.GetRealData(PlasmaIdx::Fy1).data();
        amrex::Real * const Fux1 = soa.GetRealData(PlasmaIdx::Fux1).data();
        amrex::Real * const Fuy1 = soa.GetRealData(PlasmaIdx::Fuy1).data();
        amrex::Real * const Fpsi1 = soa.GetRealData(PlasmaIdx::Fpsi1).data();
        amrex::Real * const Fx2 = soa.GetRealData(PlasmaIdx::Fx2).data();
        amrex::Real * const Fy2 = soa.GetRealData(PlasmaIdx::Fy2).data();
        amrex::Real * const Fux2 = soa.GetRealData(PlasmaIdx::Fux2).data();
        amrex::Real * const Fuy2 = soa.GetRealData(PlasmaIdx::Fuy2).data();
        amrex::Real * const Fpsi2 = soa.GetRealData(PlasmaIdx::Fpsi2).data();
        amrex::Real * const Fx3 = soa.GetRealData(PlasmaIdx::Fx3).data();
        amrex::Real * const Fy3 = soa.GetRealData(PlasmaIdx::Fy3).data();
        amrex::Real * const Fux3 = soa.GetRealData(PlasmaIdx::Fux3).data();
        amrex::Real * const Fuy3 = soa.GetRealData(PlasmaIdx::Fuy3).data();
        amrex::Real * const Fpsi3 = soa.GetRealData(PlasmaIdx::Fpsi3).data();
        amrex::Real * const Fx4 = soa.GetRealData(PlasmaIdx::Fx4).data();
        amrex::Real * const Fy4 = soa.GetRealData(PlasmaIdx::Fy4).data();
        amrex::Real * const Fux4 = soa.GetRealData(PlasmaIdx::Fux4).data();
        amrex::Real * const Fuy4 = soa.GetRealData(PlasmaIdx::Fuy4).data();
        amrex::Real * const Fpsi4 = soa.GetRealData(PlasmaIdx::Fpsi4).data();
        amrex::Real * const Fx5 = soa.GetRealData(PlasmaIdx::Fx5).data();
        amrex::Real * const Fy5 = soa.GetRealData(PlasmaIdx::Fy5).data();
        amrex::Real * const Fux5 = soa.GetRealData(PlasmaIdx::Fux5).data();
        amrex::Real * const Fuy5 = soa.GetRealData(PlasmaIdx::Fuy5).data();
        amrex::Real * const Fpsi5 = soa.GetRealData(PlasmaIdx::Fpsi5).data();

        const int depos_order_xy = Hipace::m_depos_order_xy;
        const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

        using PTileType = PlasmaParticleContainer::ParticleTileType;
        const auto getPosition = GetParticlePosition<PTileType>(pti.GetParticleTile());
        const auto SetPosition = SetParticlePosition<PTileType>(pti.GetParticleTile());
        const auto enforceBC = EnforceBC<PTileType>(pti.GetParticleTile(), lev);
        const amrex::Real zmin = xyzmin[2];
        const amrex::Real dz = dx[2];

        const amrex::Real charge = plasma.m_charge;
        const amrex::Real mass = plasma.m_mass;
        amrex::ParallelFor(pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip) {
                amrex::ParticleReal xp, yp, zp;
                int pid;
                getPosition(ip, xp, yp, zp, pid);

                if (pid < 0) return;

                // define field at particle position reals
                amrex::ParticleReal ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                amrex::ParticleReal Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

                if (do_shift)
                {
                    ShiftForceTerms(Fx1[ip], Fy1[ip], Fux1[ip], Fuy1[ip], Fpsi1[ip],
                                    Fx2[ip], Fy2[ip], Fux2[ip], Fuy2[ip], Fpsi2[ip],
                                    Fx3[ip], Fy3[ip], Fux3[ip], Fuy3[ip], Fpsi3[ip],
                                    Fx4[ip], Fy4[ip], Fux4[ip], Fuy4[ip], Fpsi4[ip],
                                    Fx5[ip], Fy5[ip], Fux5[ip], Fuy5[ip], Fpsi5[ip] );
                }

                if (do_update)
                {
                    // field gather for a single particle
                    doGatherShapeN(xp, yp, zmin,
                                   ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                                   exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                   dx_arr, xyzmin_arr, lo, depos_order_xy, 0);
                    // update force terms for a single particle
                    const amrex::Real psi_factor = phys_const.q_e/(phys_const.m_e*phys_const.c*phys_const.c);
                    UpdateForceTerms(uxp[ip], uyp[ip], psi_factor*psip[ip], ExmByp, EypBxp, Ezp,
                                     Bxp, Byp, Bzp, Fx1[ip], Fy1[ip], Fux1[ip], Fuy1[ip],
                                     Fpsi1[ip], clightsq, phys_const, charge, mass);
                }

                if (do_push)
                {
                    // push a single particle
                    PlasmaParticlePush(xp, yp, zp, uxp[ip], uyp[ip], psip[ip], x_prev[ip],
                                       y_prev[ip], ux_temp[ip], uy_temp[ip], psi_temp[ip],
                                       Fx1[ip], Fy1[ip], Fux1[ip], Fuy1[ip], Fpsi1[ip],
                                       Fx2[ip], Fy2[ip], Fux2[ip], Fuy2[ip], Fpsi2[ip],
                                       Fx3[ip], Fy3[ip], Fux3[ip], Fuy3[ip], Fpsi3[ip],
                                       Fx4[ip], Fy4[ip], Fux4[ip], Fuy4[ip], Fpsi4[ip],
                                       Fx5[ip], Fy5[ip], Fux5[ip], Fuy5[ip], Fpsi5[ip],
                                       dz, temp_slice, ip, SetPosition, enforceBC );
                }
                return;
          }
          );
      }
}

void
ResetPlasmaParticles (PlasmaParticleContainer& plasma, int const lev, const bool initial)
{
    HIPACE_PROFILE("ResetPlasmaParticles()");

    using namespace amrex::literals;

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        auto& soa = pti.GetStructOfArrays(); // For momenta and weights
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();
        amrex::Real * const x_prev = soa.GetRealData(PlasmaIdx::x_prev).data();
        amrex::Real * const y_prev = soa.GetRealData(PlasmaIdx::y_prev).data();
        amrex::Real * const ux_temp = soa.GetRealData(PlasmaIdx::ux_temp).data();
        amrex::Real * const uy_temp = soa.GetRealData(PlasmaIdx::uy_temp).data();
        amrex::Real * const psi_temp = soa.GetRealData(PlasmaIdx::psi_temp).data();
        amrex::Real * const Fx1 = soa.GetRealData(PlasmaIdx::Fx1).data();
        amrex::Real * const Fy1 = soa.GetRealData(PlasmaIdx::Fy1).data();
        amrex::Real * const Fux1 = soa.GetRealData(PlasmaIdx::Fux1).data();
        amrex::Real * const Fuy1 = soa.GetRealData(PlasmaIdx::Fuy1).data();
        amrex::Real * const Fpsi1 = soa.GetRealData(PlasmaIdx::Fpsi1).data();
        amrex::Real * const Fx2 = soa.GetRealData(PlasmaIdx::Fx2).data();
        amrex::Real * const Fy2 = soa.GetRealData(PlasmaIdx::Fy2).data();
        amrex::Real * const Fux2 = soa.GetRealData(PlasmaIdx::Fux2).data();
        amrex::Real * const Fuy2 = soa.GetRealData(PlasmaIdx::Fuy2).data();
        amrex::Real * const Fpsi2 = soa.GetRealData(PlasmaIdx::Fpsi2).data();
        amrex::Real * const Fx3 = soa.GetRealData(PlasmaIdx::Fx3).data();
        amrex::Real * const Fy3 = soa.GetRealData(PlasmaIdx::Fy3).data();
        amrex::Real * const Fux3 = soa.GetRealData(PlasmaIdx::Fux3).data();
        amrex::Real * const Fuy3 = soa.GetRealData(PlasmaIdx::Fuy3).data();
        amrex::Real * const Fpsi3 = soa.GetRealData(PlasmaIdx::Fpsi3).data();
        amrex::Real * const Fx4 = soa.GetRealData(PlasmaIdx::Fx4).data();
        amrex::Real * const Fy4 = soa.GetRealData(PlasmaIdx::Fy4).data();
        amrex::Real * const Fux4 = soa.GetRealData(PlasmaIdx::Fux4).data();
        amrex::Real * const Fuy4 = soa.GetRealData(PlasmaIdx::Fuy4).data();
        amrex::Real * const Fpsi4 = soa.GetRealData(PlasmaIdx::Fpsi4).data();
        amrex::Real * const Fx5 = soa.GetRealData(PlasmaIdx::Fx5).data();
        amrex::Real * const Fy5 = soa.GetRealData(PlasmaIdx::Fy5).data();
        amrex::Real * const Fux5 = soa.GetRealData(PlasmaIdx::Fux5).data();
        amrex::Real * const Fuy5 = soa.GetRealData(PlasmaIdx::Fuy5).data();
        amrex::Real * const Fpsi5 = soa.GetRealData(PlasmaIdx::Fpsi5).data();
        amrex::Real * const x0 = soa.GetRealData(PlasmaIdx::x0).data();
        amrex::Real * const y0 = soa.GetRealData(PlasmaIdx::y0).data();
        amrex::Real * const w = soa.GetRealData(PlasmaIdx::w).data();
        amrex::Real * const w0 = soa.GetRealData(PlasmaIdx::w0).data();

        const auto GetPosition =
            GetParticlePosition<PlasmaParticleContainer::ParticleTileType>(pti.GetParticleTile());
        const auto SetPosition =
            SetParticlePosition<PlasmaParticleContainer::ParticleTileType>(pti.GetParticleTile());

        amrex::ParallelFor(pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip) {

                amrex::ParticleReal xp, yp, zp;
                int pid;
                GetPosition(ip, xp, yp, zp, pid);
                if (initial == false){
                    SetPosition(ip, x_prev[ip], y_prev[ip], zp);
                } else {
                    SetPosition(ip, x0[ip], y0[ip], zp, std::abs(pid));
                    w[ip] = w0[ip];
                    uxp[ip] = 0._rt;
                    uyp[ip] = 0._rt;
                    psip[ip] = 0._rt;
                    x_prev[ip] = 0._rt;
                    y_prev[ip] = 0._rt;
                    ux_temp[ip] = 0._rt;
                    uy_temp[ip] = 0._rt;
                    psi_temp[ip] = 0._rt;
                    Fx1[ip] = 0._rt;
                    Fy1[ip] = 0._rt;
                    Fux1[ip] = 0._rt;
                    Fuy1[ip] = 0._rt;
                    Fpsi1[ip] = 0._rt;
                    Fx2[ip] = 0._rt;
                    Fy2[ip] = 0._rt;
                    Fux2[ip] = 0._rt;
                    Fuy2[ip] = 0._rt;
                    Fpsi2[ip] = 0._rt;
                    Fx3[ip] = 0._rt;
                    Fy3[ip] = 0._rt;
                    Fux3[ip] = 0._rt;
                    Fuy3[ip] = 0._rt;
                    Fpsi3[ip] = 0._rt;
                    Fx4[ip] = 0._rt;
                    Fy4[ip] = 0._rt;
                    Fux4[ip] = 0._rt;
                    Fuy4[ip] = 0._rt;
                    Fpsi4[ip] = 0._rt;
                    Fx5[ip] = 0._rt;
                    Fy5[ip] = 0._rt;
                    Fux5[ip] = 0._rt;
                    Fuy5[ip] = 0._rt;
                    Fpsi5[ip] = 0._rt;
                }
        }
        );
    }
}
