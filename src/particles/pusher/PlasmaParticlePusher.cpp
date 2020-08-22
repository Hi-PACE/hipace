#include "PlasmaParticlePusher.H"

#include "particles/PlasmaParticleContainer.H"
#include "FieldGather.H"
#include "UpdateForceTerms.H"
#include "fields/Fields.H"
#include "Constants.H"
#include "Hipace.H"
#include "GetAndSetPosition.H"

void
UpdateForcePushParticles (PlasmaParticleContainer& plasma, Fields & fields,
                          amrex::Geometry const& gm, int const lev)
{
    BL_PROFILE("UpdateForcePushParticles_PlasmaParticleContainer()");
    using namespace amrex::literals;

    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();
    const PhysConst phys_const = get_phys_const();

    const amrex::MultiFab& S = fields.getSlices(lev, 1);

    for ( amrex::MFIter mfi(S); mfi.isValid(); ++mfi )
    {
        // Extract properties associated with the extent of the current box
        // Grow to capture the extent of the particle shape
        amrex::Box tilebox = mfi.tilebox().grow(
            {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);
        // Extract the fields
        const amrex::MultiFab exmby(S, amrex::make_alias, FieldComps::ExmBy, 1);
        const amrex::MultiFab eypbx(S, amrex::make_alias, FieldComps::EypBx, 1);
        const amrex::MultiFab ez(S, amrex::make_alias, FieldComps::Ez, 1);
        const amrex::MultiFab bx(S, amrex::make_alias, FieldComps::Bx, 1);
        const amrex::MultiFab by(S, amrex::make_alias, FieldComps::By, 1);
        const amrex::MultiFab bz(S, amrex::make_alias, FieldComps::Bz, 1);
        // Extract FabArray for this box
        const amrex::FArrayBox& exmby_fab = exmby[mfi];
        const amrex::FArrayBox& eypbx_fab = eypbx[mfi];
        const amrex::FArrayBox& ez_fab = ez[mfi];
        const amrex::FArrayBox& bx_fab = bx[mfi];
        const amrex::FArrayBox& by_fab = by[mfi];
        const amrex::FArrayBox& bz_fab = bz[mfi];
        // Extract field array from FabArray
        amrex::Array4<const amrex::Real> const& exmby_arr = exmby_fab.array();
        amrex::Array4<const amrex::Real> const& eypbx_arr = eypbx_fab.array();
        amrex::Array4<const amrex::Real> const& ez_arr = ez_fab.array();
        amrex::Array4<const amrex::Real> const& bx_arr = bx_fab.array();
        amrex::Array4<const amrex::Real> const& by_arr = by_fab.array();
        amrex::Array4<const amrex::Real> const& bz_arr = bz_fab.array();

        const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
        const amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};
        auto& particles = plasma.GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& soa = particle_tile.GetStructOfArrays();

        // loading the data
        amrex::Real * const uxp = soa.GetRealData(PlasmaIdx::ux).data();
        amrex::Real * const uyp = soa.GetRealData(PlasmaIdx::uy).data();
        amrex::Real * const psip = soa.GetRealData(PlasmaIdx::psi).data();

        amrex::Real * const Fx1 = soa.GetRealData(PlasmaIdx::Fx1).data();
        amrex::Real * const Fy1 = soa.GetRealData(PlasmaIdx::Fy1).data();
        amrex::Real * const Fux1 = soa.GetRealData(PlasmaIdx::Fux1).data();
        amrex::Real * const Fuy1 = soa.GetRealData(PlasmaIdx::Fuy1).data();
        amrex::Real * const Fpsi1 = soa.GetRealData(PlasmaIdx::Fpsi1).data();

        const int depos_order_xy = Hipace::m_depos_order_xy;
        const amrex::Real clightsq = 1.0_rt/(phys_const.c*phys_const.c);

        const amrex::Real zmin = xyzmin[2];

        PlasmaParticleContainer::ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();

        amrex::ParallelFor(particle_tile.GetArrayOfStructs().size(),
            [=] AMREX_GPU_DEVICE (long ip) {
                PlasmaParticleContainer::ParticleType& p = pstruct[ip];
                amrex::ParticleReal xp = p.pos(0);
                amrex::ParticleReal yp = p.pos(1);
                amrex::ParticleReal ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
                amrex::ParticleReal Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

                // field gather for a single particle
                doGatherShapeN(xp, yp, zmin,
                    ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                    exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                    dx_arr, xyzmin_arr, lo, depos_order_xy, 0);

                // update force terms for a single particle
                UpdateForceTerms( uxp[ip], uyp[ip], psip[ip], ExmByp, EypBxp, Ezp,
                                  Bxp, Byp, Bzp, Fx1[ip], Fy1[ip], Fux1[ip], Fuy1[ip],
                                  Fpsi1[ip], clightsq);

                //insert push a single particle
          }
          );
      }
}
