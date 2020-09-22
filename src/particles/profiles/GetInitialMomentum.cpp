#include "GetInitialMomentum.H"
#include <AMReX_ParmParse.H>

GetInitialMomentum::GetInitialMomentum ()
{
    amrex::ParmParse pp("beam");
    std::string profile;
    pp.get("momentum_profile", profile);
    if        (profile == "gaussian") {
        m_momentum_profile = BeamMomentumType::Gaussian;
    } else {
        amrex::Abort("Unknown beam momentum profile!");
    }

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
    }
}
