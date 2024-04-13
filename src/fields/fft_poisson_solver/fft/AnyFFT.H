/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
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
    C2R_1D_batched,
    R2C_1D_batched
};

struct AnyFFT {

    std::size_t Initialize (FFTType type, int nx, int ny);

    void SetBuffers (void* in, void* out, void* work_area);

    void Execute ();

    ~AnyFFT ();

private:
    VendorPlan* m_plan = nullptr;
};

#endif // ANYFFT_H_
