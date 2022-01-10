#ifndef LASER_H_
#define LASER_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>

/** \brief describes which slice with respect to the currently calculated is used */
struct WhichLaserSlice {
    enum slice { This=0, PrevZeta1, PrevZeta2, PrevTime1, PrevTime2, AbsSq, AbsSqDx, AbsSqDy, N };
};

class Laser
{
private:
        static constexpr int m_nslices = 8; /**< 0: use laser envelope. 1: use E and B fields */
public:
    /** Constructor */
    explicit Laser () //std::string name) :         // m_name(name)
    {
        ReadParameters();
    };

    void ReadParameters ();

    /** get function for the 2D slices
     * \param[in] islice slice index
     */
    amrex::MultiFab& getSlices (int islice) {return m_slices[islice]; }

    /** \brief Allocate beam particle data and initialize particles with requested beam profile
     * \return physical time at which the simulation will start
     */
    void InitData (const amrex::BoxArray& slice_ba,
                   const amrex::DistributionMapping& slice_dm);

    void PrepareLaserSlice (const amrex::Geometry& geom, const int islice);

    // Same function whether the full fields are specified or just the envelope.
    // Just somewhere: if (m_EB) { F *= cos (k0 * z) }

    /** Number of guard cells for slices MultiFab */
    static amrex::IntVect m_slices_nguards;
private:
    std::string m_name = "laser"; /**< name of the laser */
    amrex::Real m_a0 {0.}; /**< Laser peak normalized amplitude */
    amrex::Real m_w0 {0.}; /**< Laser waist */
    amrex::Real m_L0 {0.}; /**< Laser length (HW 1/e in amplitude) */
    amrex::Real m_lambda0 {0.}; /**< Laser central wavelength */
    /** Average position of the Gaussian laser pulse */
    amrex::RealVect m_position_mean {0., 0., 0.};
    /** Width of the Gaussian laser pulse */
    amrex::RealVect m_position_std {0., 0., 0.};
    int m_EB; /**< 0: use laser envelope. 1: use E and B fields */

    /** Vector over levels, all fields */
    // amrex::Vector<amrex::FArrayBox> m_F;
    amrex::Vector<std::array<amrex::FArrayBox, 3>> m_3D_F; // used to be amrex::MultiFab
    /** Vector over levels, array of 4 slices required to compute current slice */
    std::array<amrex::MultiFab, m_nslices> m_slices;
};

#endif // LASER_H_
