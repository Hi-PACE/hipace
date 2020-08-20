#include "PlasmaParticlePusher.H"

#include "particles/PlasmaParticleContainer.H"
#include "FieldGather.H"
#include "UpdateForceTerms.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"


void
UpdateForcePushParticles (PlasmaParticleContainer& plasma, Fields & fields,
                          amrex::Geometry const& gm, int const lev)
{
    BL_PROFILE("UpdateForcePushParticles_PlasmaParticleContainer()");
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
        const amrex::MultiFab& S = fields.getSlices(lev, 1);
        const amrex::MultiFab exmby(S, amrex::make_alias, FieldComps::ExmBy, 1);
        const amrex::MultiFab eypbx(S, amrex::make_alias, FieldComps::EypBx, 1);
        const amrex::MultiFab ez(S, amrex::make_alias, FieldComps::Ez, 1);
        const amrex::MultiFab bx(S, amrex::make_alias, FieldComps::Bx, 1);
        const amrex::MultiFab by(S, amrex::make_alias, FieldComps::By, 1);
        const amrex::MultiFab bz(S, amrex::make_alias, FieldComps::Bz, 1);
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

        const auto& aos = pti.GetArrayOfStructs(); // For positions
        const auto& pos_structs = aos.begin();
        auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        // loading the data
        amrex::Real * uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * psip = soa.GetRealData(PlasmaIdx::psi).data();

        amrex::Real * Fx1 = soa.GetRealData(PlasmaIdx::Fx1).data();
        amrex::Real * Fy1 = soa.GetRealData(PlasmaIdx::Fy1).data();
        amrex::Real * Fux1 = soa.GetRealData(PlasmaIdx::Fux1).data();
        amrex::Real * Fuy1 = soa.GetRealData(PlasmaIdx::Fuy1).data();
        amrex::Real * Fpsi1 = soa.GetRealData(PlasmaIdx::Fpsi1).data();

        const int depos_order_xy = Hipace::m_depos_order_xy;
        const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

        amrex::ParallelFor(pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip) {

                // define field at particle position reals
                amrex::ParticleReal ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                amrex::ParticleReal Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

                // field gather for a single particle
                doGatherShapeN(pos_structs[ip].pos(0), pos_structs[ip].pos(1), xyzmin[2],
                    ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                    exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                    dx_arr, xyzmin_arr, lo, depos_order_xy, 0);

                // insert update force terms for a single particle
                const amrex::Real gammap = (1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                   + uyp[ip]*uyp[ip]*clightsq
                                                   + psip[ip]*psip[ip])/(2.0_rt * psip[ip] );

                const amrex::Real charge_mass_ratio = -1.0_rt;

                /* Change for x-position along zeta */
                Fx1[ip] = uxp[ip] / psip[ip];
                /* Change for y-position along zeta */
                Fy1[ip] = -uyp[ip] / psip[ip];
                /* Change for ux along zeta */
                Fux1[ip] = -charge_mass_ratio * ( gammap * ExmByp / psip[ip] + Byp + ( uyp[ip] * Bzp ) / psip[ip] );
                /* Change for uy along zeta */
                Fuy1[ip] = -charge_mass_ratio * ( gammap * EypBxp / psip[ip] - Bxp - ( uxp[ip] * Bzp ) / psip[ip] );
                /* Change for psi along zeta */
                Fpsi1[ip] = -charge_mass_ratio * (( uxp[ip] * ExmByp + uyp[ip] * EypBxp ) / psip[ip] - Ezp );

                //insert push a single particle

          }
          );
      }
}
