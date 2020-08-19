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
    // Extract properties associated with physical size of the box
    amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

    PhysConst phys_const = get_phys_const();

    // Loop over particle boxes
    for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
    {
        // Extract properties associated with the extent of the current box
        amrex::Box tilebox = pti.tilebox().grow(2); // Grow to capture the extent of the particle shape

        amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
        amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
        amrex::Dim3 const lo = amrex::lbound(tilebox);

        // Extract the fields currents
        // amrex::MultiFab& S = fields.getSlices(lev, 1);
        // amrex::MultiFab jx(S, amrex::make_alias, FieldComps::jx, S.nGrow());
        // amrex::MultiFab jy(S, amrex::make_alias, FieldComps::jy, S.nGrow());
        // amrex::MultiFab jz(S, amrex::make_alias, FieldComps::jz, S.nGrow());
        // // Extract FabArray for this box
        // amrex::FArrayBox& jx_fab = jx[pti];
        // amrex::FArrayBox& jy_fab = jy[pti];
        // amrex::FArrayBox& jz_fab = jz[pti];
      }
}

// void
// UpdateForceTerms (PlasmaParticleContainer& plasma, Fields & fields,
//                 amrex::Geometry const& gm, int const lev)
// {
//     BL_PROFILE("DepositCurrent_PlasmaParticleContainer()");
//     // Extract properties associated with physical size of the box
//     amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();
//
//     PhysConst phys_const = get_phys_const();
//
//     // Loop over particle boxes
//     for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
//     {
//         // Extract properties associated with the extent of the current box
//         amrex::Box tilebox = pti.tilebox().grow(2); // Grow to capture the extent of the particle shape
//
//         amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
//         amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
//         amrex::Dim3 const lo = amrex::lbound(tilebox);
//
//         // Extract the fields currents
//         // amrex::MultiFab& S = fields.getSlices(lev, 1);
//         // amrex::MultiFab jx(S, amrex::make_alias, FieldComps::jx, S.nGrow());
//         // amrex::MultiFab jy(S, amrex::make_alias, FieldComps::jy, S.nGrow());
//         // amrex::MultiFab jz(S, amrex::make_alias, FieldComps::jz, S.nGrow());
//         // // Extract FabArray for this box
//         // amrex::FArrayBox& jx_fab = jx[pti];
//         // amrex::FArrayBox& jy_fab = jy[pti];
//         // amrex::FArrayBox& jz_fab = jz[pti];
//       }
// }


// void
// PlasmaParticlePusher (PlasmaParticleContainer& plasma, //Fields & fields,
//                 amrex::Geometry const& gm, int const lev)
// {
//     BL_PROFILE("DepositCurrent_PlasmaParticleContainer()");
//     // Extract properties associated with physical size of the box
//     amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();
//
//     PhysConst phys_const = get_phys_const();
//
//     // Loop over particle boxes
//     for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
//     {
//         // Extract properties associated with the extent of the current box
//         amrex::Box tilebox = pti.tilebox().grow(2); // Grow to capture the extent of the particle shape
//
//         amrex::RealBox const grid_box{tilebox, gm.CellSize(), gm.ProbLo()};
//         amrex::Real const * AMREX_RESTRICT xyzmin = grid_box.lo();
//         amrex::Dim3 const lo = amrex::lbound(tilebox);
//
//         // // Extract the fields currents
//         // amrex::MultiFab& S = fields.getSlices(lev, 1);
//         // amrex::MultiFab jx(S, amrex::make_alias, FieldComps::jx, S.nGrow());
//         // amrex::MultiFab jy(S, amrex::make_alias, FieldComps::jy, S.nGrow());
//         // amrex::MultiFab jz(S, amrex::make_alias, FieldComps::jz, S.nGrow());
//         // // Extract FabArray for this box
//         // amrex::FArrayBox& jx_fab = jx[pti];
//         // amrex::FArrayBox& jy_fab = jy[pti];
//         // amrex::FArrayBox& jz_fab = jz[pti];
//       }
// }
