#ifndef LASER_H_
#define LASER_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>

/** \brief describes which slice with respect to the currently calculated is used */
struct WhichLaserSlice {
    enum slice { This=0, PrevZeta1, PrevZeta2, PrevTime1, PrevTime2, N };
};

class Laser
{
private:
        static constexpr int m_nslices = 5;
public:
    /** Constructor */
    explicit Laser ()
    {
        ReadParameters();
    }

    void ReadParameters ();

    /** get function for the 2D slices
     * \param[in] islice slice index */
    amrex::MultiFab& getSlices (int islice) {return m_slices[islice]; }
    /** get function for the 2D slices (const version)
     * \param[in] islice slice index */
    const amrex::MultiFab& getSlices (int islice) const {return m_slices[islice]; }

    /** \brief Allocate beam particle data and initialize particles with requested beam profile
     * \return physical time at which the simulation will start
     */
    void InitData (const amrex::BoxArray& slice_ba,
                   const amrex::DistributionMapping& slice_dm);

    void PrepareLaserSlice (const amrex::Geometry& geom, const int islice);

    bool m_use_laser {false}; /**< whether a laser is used or not */

private:
    std::string m_name = "laser"; /**< name of the laser */
    amrex::Real m_a0 {0.}; /**< Laser peak normalized amplitude */
    amrex::RealVect m_w0 {0., 0., 0.}; /**< Laser waist in x and y (the third value is omitted) */
    amrex::Real m_L0 {0.}; /**< Laser length (HW 1/e in amplitude) */
    amrex::Real m_tau {0.}; /**< Laser duration (HW 1/e in amplitude) */
    amrex::Real m_lambda0 {0.}; /**< Laser central wavelength */
    /** Average position of the Gaussian laser pulse */
    amrex::RealVect m_position_mean {0., 0., 0.};

    /** Number of guard cells for slices MultiFab */
    amrex::IntVect m_slices_nguards = {-1, -1, -1};

    /** Vector over levels, all fields */
    amrex::Vector<std::array<amrex::FArrayBox, 3>> m_3D_F;
    /** Vector over levels, array of 4 slices required to compute current slice */
    std::array<amrex::MultiFab, m_nslices> m_slices;
};

#endif // LASER_H_
