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

    // const PhysConst phys_const = get_phys_const();

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        amrex::Box tilebox = pti.tilebox().grow(2); // Grow to capture the extent of the particle shape

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields
        amrex::MultiFab& S = fields.getSlices(lev, 1);
        amrex::MultiFab exmby(S, amrex::make_alias, FieldComps::ExmBy, 1);
        amrex::MultiFab eypbx(S, amrex::make_alias, FieldComps::EypBx, 1);
        amrex::MultiFab ez(S, amrex::make_alias, FieldComps::Ez, 1);
        amrex::MultiFab bx(S, amrex::make_alias, FieldComps::Bx, 1);
        amrex::MultiFab by(S, amrex::make_alias, FieldComps::By, 1);
        amrex::MultiFab bz(S, amrex::make_alias, FieldComps::Bz, 1);
        // Extract FabArray for this box
        amrex::FArrayBox& exmby_fab = exmby[pti];
        amrex::FArrayBox& eypbx_fab = eypbx[pti];
        amrex::FArrayBox& ez_fab = ez[pti];
        amrex::FArrayBox& bx_fab = bx[pti];
        amrex::FArrayBox& by_fab = by[pti];
        amrex::FArrayBox& bz_fab = bz[pti];
        // Extract field array from FabArray
        amrex::Array4<const amrex::Real> const& exmby_arr = exmby_fab.array();
        amrex::Array4<const amrex::Real> const& eypbx_arr = eypbx_fab.array();
        amrex::Array4<const amrex::Real> const& ez_arr = ez_fab.array();
        amrex::Array4<const amrex::Real> const& bx_arr = bx_fab.array();
        amrex::Array4<const amrex::Real> const& by_arr = by_fab.array();
        amrex::Array4<const amrex::Real> const& bz_arr = bz_fab.array();

        amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
        amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};

        const auto& aos = pti.GetArrayOfStructs(); // For positions
        const auto& pos_structs = aos.begin();
        // const auto& soa = pti.GetStructOfArrays(); // For momenta and weights

        amrex::ParallelFor(
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip) {

              // define field at particle position reals
              amrex::ParticleReal ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
              amrex::ParticleReal Bxp = 0._rt, Byp = 0._rt, Bzp = 0._rt;

              // field gather for a single particle
              doGatherShapeN(pos_structs[ip].pos(0), pos_structs[ip].pos(1), xyzmin[2],
                              ExmByp, EypBxp, Ezp, Bxp, Byp, Bzp,
                              exmby_arr, eypbx_arr, ez_arr, bx_arr, by_arr, bz_arr,
                              dx_arr, xyzmin_arr, lo, Hipace::m_depos_order_xy, 0);

              if (abs(pos_structs[ip].pos(0)) < 0.63 &&  abs(pos_structs[ip].pos(1)) < 5 )
              {
                std::cout <<"x pos " << pos_structs[ip].pos(0) <<  " y pos " << pos_structs[ip].pos(1) <<  " Bx: " << Bxp << " By: " << Byp  <<"\n";
              }

              // insert update force terms for a single particle

              //insert push a single particle

          }
          );


      }
}
