#include "GetInitialMomentum.H"

GetInitialMomentum::GetInitialMomentum (amrex::Real profile, amrex::Real mean, amrex::Real std)
    : m_profile(profile), m_mean(mean), m_std(std)
{
    pp = ParmParse("beam");
    if        (m_profile == "gaussian") {
        pp.get("mean", m_mean);
        pp.query("std", m_std);
    } else {
        AMREX_ABORT("unknown profile!");
    }
}

void GetInitialMomentum::operator() (amrex::Real& u, amrex::Real x, amrex::Real y, amrex::z)
{
    if (m_profile == "gaussian"){
        ParticleUtil::get_gaussian_random_momentum(u, m_mean, m_std);
    }

    return u;
}

