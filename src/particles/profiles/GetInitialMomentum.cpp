#include "GetInitialMomentum.H"
#include <AMReX_ParmParse.H>

GetInitialMomentum::GetInitialMomentum ()
{
    amrex::ParmParse pp("beam");

    /* currently only Gaussian beam momentum profile implemented */
    if (m_momentum_profile == BeamMomentumType::Gaussian) {

        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        if (pp.query("u_mean", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_u_mean[idim] = loc_array[idim];
            }
        }
        if (pp.query("u_std", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_u_std[idim] = loc_array[idim];
            }
        }
        bool do_symmetrize;
        pp.query("do_symmetrize", do_symmetrize);
        if (do_symmetrize) AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_u_mean[0]+m_u_mean[1] < 1e-12,
        "Symmetrizing the beam is only implemented for no mean momentum in x and y");
    } else {
        amrex::Abort("Unknown beam momentum profile!");
    }
}
