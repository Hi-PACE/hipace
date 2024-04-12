/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#ifndef ANYDST_H_
#define ANYDST_H_

#include "AnyFFT.H"

#include <AMReX_GpuComplex.H>
#include <AMReX_FArrayBox.H>

#include <memory>

/**
 * \brief Wrapper around multiple FFT libraries.
 *
 * The header file defines the API and the base types
 * (Complex and VendorFFTPlan), and the implementation for different FFT libraries is
 * done in different cpp files. This wrapper only depends on the underlying FFT library
 * AND on AMReX (There is no dependence on WarpX).
 */
namespace AnyDST
{

    /** Direction in which the FFT is performed. */
    enum struct direction {forward, backward};

    /** \brief This struct contains the vendor FFT plan and additional metadata
     */
    struct DSTplan
    {

        /** pointer to array in position space */
        amrex::FArrayBox* m_position_array;
        /** pointer to array in Fourier space */
        amrex::FArrayBox* m_fourier_array;

        /** Expanded data in position space, only for Cuda */
        std::unique_ptr<amrex::FArrayBox> m_expanded_position_array;
        /** Expanded data in Fourier, only for Cuda */
        std::unique_ptr<amrex::BaseFab<amrex::GpuComplex<amrex::Real>>> m_expanded_fourier_array;

        /** Vendor FFT plan */
        AnyFFT::VendorFFTPlan m_plan;
        /** Vendor FFT plan for the transpose transform. Used for use_small_dst and DSTW */
        AnyFFT::VendorFFTPlan m_plan_b;

        /** Use large R2C or small C2R dst */
        bool use_small_dst;

#if defined(AMREX_USE_HIP)
        /** execution info for rocFFT */
        rocfft_execution_info m_execinfo;
        /** work buffer for rocFFT */
        void* m_buffer;
#endif
    };

    /** Collection of FFT plans, one FFTplan per box */
    using DSTplans = amrex::LayoutData<DSTplan>;

    /** \brief create FFT plan for the backend FFT library.
     * \param[in] real_size Size of the real array, along each dimension.
     * \param[out] position_array Real array from/to where R2R DST is performed
     * \param[out] fourier_array Real array to/from where R2R DST is performed
     */
    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array);

    /** \brief Destroy library FFT plan.
     * \param[out] dst_plan plan to destroy
     */
    void DestroyPlan (DSTplan& dst_plan);

    /** \brief Perform FFT with backend library.
     * \param[out] dst_plan plan for which the FFT is performed
     * \param[in] d direction of the FFT
     */
    void Execute (DSTplan& dst_plan, AnyDST::direction d);
}

#endif // ANYDST_H_
