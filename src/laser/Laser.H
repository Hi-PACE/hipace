#ifndef LASER_H_
#define LASER_H_

#include "fields/Fields.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrCore.H>

/** \brief describes which slice with respect to the currently calculated is used */
struct WhichLaserSlice {
    enum slice { This=0, PrevZeta1, PrevZeta2, PrevTime1, PrevTime2, NextTime, N };
};

class Fields;

class Laser
{
private:
    static constexpr int m_nslices = WhichLaserSlice::N;
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
    const amrex::FArrayBox& getFAB () const {return m_F;};
    amrex::FArrayBox& getFAB () {return m_F;};

    /** \brief Allocate beam particle data and initialize particles with requested beam profile
     * \return physical time at which the simulation will start
     */
    void InitData (const amrex::BoxArray& slice_ba,
                   const amrex::DistributionMapping& slice_dm);

    /** \brief Initialize 3D laser field on current box.
     *
     * \param[in] step time step of the simulation
     * \param[in] bx Box on which the laser field is initialized
     * \param[in] gm Geometry of the problem
     */
    void Init3DEnvelope (int step, amrex::Box bx, const amrex::Geometry& gm);

    /** \brief Copy from 2D slice on device to 3D array on host, and vice-versa
     *
     * \param[in] isl slice index, referring to the 3D slice
     * \param[in] to3d if true, copy from 2D slice to 3D array. Otherwise, the other way.
     */
    void Copy (int isl, bool to3d);

    void AdvanceSlice (const Fields& fields);

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

    /** Nb fields in 3D array: A_new_real, A_new_imag, A_old_real, A_old_imag */
    int m_nfields_3d {4};
    /** 3D laser data. Could be a vector over levels if MR needed for laser */
    amrex::FArrayBox m_F;
    /** Array of N slices required to compute current slice */
    std::array<amrex::MultiFab, m_nslices> m_slices;
};

#endif // LASER_H_
