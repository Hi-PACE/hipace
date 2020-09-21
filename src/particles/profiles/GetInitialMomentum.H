#ifndef GETINTIALMOMENTUM_H_
#define GETINTIALMOMENTUM_H_

struct GetInitialMomentum
{
    GetInitialMomentum () = default;

    GetInitialMomentum (amrex::Real profile, amrex::Real mean, amrex::Real std);

    operator() (amrex::Real[3] u, amrex::Real x, amrex::Real y, amrex::Real z);
    amrex::Real[3] m_mean;
    amrex::Real[3] m_std {0.,0.,0.};
    amrex::Real m_profile;
}

#endif // GETINTIALMOMENTUM_H_
