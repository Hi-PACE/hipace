/* Copyright 2020-2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Remi Lehe, Severin Diederichs
 * WeiqunZhang
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ANYFFT_H_
#define ANYFFT_H_

#include <cstddef>

struct VendorPlan;

enum struct FFTType {
    C2C_2D_fwd,
    C2C_2D_bkw,
    C2R_2D,
    R2C_2D,
    R2R_2D,
    C2R_1D_batched
};

struct AnyFFT {

    /** \brief Initialize an FFT plan for the requested transform type using a Vendor FFT library.
     * For 1D batched transforms, ny represents the number of batches to calculate at once.
     * This function returns the number of bytes of the work area needed for the FFT. The work area
     * has to be allocated by the function that uses the FFT and passed into SetBuffers.
     *
     * \param[in] type Type of FFT to perform
     * \param[in] nx Size of the contiguous dimension of the FFT
     * \param[in] ny Size of the second dimension of the FFT
     */
    std::size_t Initialize (FFTType type, int nx, int ny);

    /** \brief Set the pointers to the input, output and work area of the FFT.
     * This function has to be called after Initialize and before Execute.
     *
     * \param[in] in Pointer to the input of the FFT
     * \param[in] out Pointer to the output of the FFT
     * \param[in] work_area Pointer to the work area for the FFT
     */
    void SetBuffers (void* in, void* out, void* work_area);

    /** \brief Perform the initialized FFT */
    void Execute ();

    /** \brief Destructor to destroy the FFT plan */
    ~AnyFFT ();

    /** \brief Setup function that has to be called before any FFT plan is initialized. */
    static void setup ();

    /** \brief Cleanup function that has to be called at the end of the program. */
    static void cleanup ();

private:
    /** Vendor specific data for the FFT */
    VendorPlan* m_plan = nullptr;
};

#endif // ANYFFT_H_
