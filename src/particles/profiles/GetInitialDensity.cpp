#include "GetInitialDensity.H"

#include <AMReX_ParmParse.H>

GetInitialDensity::GetInitialDensity (amrex::Real a_density) //amrex::Real profile, amrex::Real mean, amrex::Real std)
    //: m_profile(profile), m_mean(mean), m_std(std)
{
    m_density = a_density;
    amrex::ParmParse pp("beam");
    std::string profile;
    pp.get("profile", profile); // to be switched to query later and default to Gaussian
    if        (profile == "gaussian") {
        m_profile = BeamProfileType::Gaussian;
    } else if (profile == "flattop") {
        m_profile = BeamProfileType::Flattop;
    } else {
        amrex::Abort("Unknown beam profile!");
    }

    if        (m_profile == BeamProfileType::Gaussian) {
        // pp.get("mean", m_mean);
        // pp.query("std", m_std);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        if (pp.query("mean", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_mean[idim] = loc_array[idim];
            }
        }
        if (pp.query("std", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_std[idim] = loc_array[idim];
            }
        }
    } else if (m_profile == BeamProfileType::Flattop) {
    } else {
        amrex::Abort("unknown profile!");
    }
}

// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// amrex::Real GetInitialDensity::operator() (amrex::Real x, amrex::Real y, amrex::Real z) const
// {
//     using namespace amrex::literals;
//     amrex::Real weight = 0._rt;
//     if        (m_profile == BeamProfileType::Gaussian){
//         weight = 0._rt; // to be Gaussian
//     } else if (m_profile == BeamProfileType::Flattop)
//         weight = m_density; // scale factor for SI units missing!
//     return weight;
// }
