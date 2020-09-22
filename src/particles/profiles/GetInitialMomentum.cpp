#include "GetInitialMomentum.H"


#include <AMReX_ParmParse.H>

GetInitialMomentum::GetInitialMomentum () //amrex::Real profile, amrex::Real mean, amrex::Real std)
    //: m_profile(profile), m_mean(mean), m_std(std)
{
    amrex::ParmParse pp("beam");
    std::string profile;
    pp.get("momentum_profile", profile); // to be switched to query later and default to Gaussian
    if        (profile == "gaussian") {
        m_momentum_profile = BeamMomentumType::Gaussian;
    } else {
        amrex::Abort("Unknown beam momentum profile!");
    }

    if (m_momentum_profile == BeamMomentumType::Gaussian) {
        // pp.get("u_mean", m_u_mean);
        // pp.query("u_std", m_u_std);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> loc_array;
        if (pp.query("u_mean", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_u_mean[idim] = loc_array[idim];
            }
        }
        std::cout << "after read in m_u_mean[0] " << m_u_mean[0] << " m_u_mean[1] " << m_u_mean[1] << " m_u_mean[2] " << m_u_mean[2] << "\n"; 
        if (pp.query("u_std", loc_array)) {
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                m_u_std[idim] = loc_array[idim];
            }
        }
    } else {
        amrex::Abort("Unknown beam momentum profile!");
    }
}

// void GetInitialMomentum::operator() (amrex::Real& ux, amrex::Real& uy, amrex::Real& uz ) const //, amrex::Real x, amrex::Real y, amrex::z)
// {
//     amrex::Real u[3] = {ux,uy,uz};
//     if (m_momentum_profile == BeamMomentumType::Gaussian){
//         ParticleUtil::get_gaussian_random_momentum(u, m_u_mean, m_u_std);
//     }
//
// }
