#include "GetInitialDensity.H"

#include <AMReX_ParmParse.H>

GetInitialDensity::GetInitialDensity (amrex::Real a_density)
{
    m_density = a_density;
    amrex::ParmParse pp("beam");
    std::string profile;
    pp.get("profile", profile);
    if        (profile == "gaussian") {
        m_profile = BeamProfileType::Gaussian;
    } else if (profile == "flattop") {
        m_profile = BeamProfileType::Flattop;
    } else {
        amrex::Abort("Unknown beam profile");
    }

    if (m_profile == BeamProfileType::Gaussian) {
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        if (pp.query("mean", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_mean[idim] = loc_array[idim];
            }
        }
        if (pp.get("std", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_std[idim] = loc_array[idim];
            }
        }
    } else if (m_profile == BeamProfileType::Flattop) {} // nothing to be done here yet
    }
}
