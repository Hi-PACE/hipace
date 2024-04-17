/* Copyright 2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 *
 * License: BSD-3-Clause-LBNL
 */
#include "AnyFFT.H"

#include <AMReX.H>
#include <AMReX_GpuDevice.H>

#include <cufftXt.h>

#include <map>
#include <string>

#ifdef AMREX_USE_FLOAT
static constexpr bool use_float = true;
#else
static constexpr bool use_float = false;
#endif

struct VendorPlan {
    cufftHandle m_cufftplan;
    int m_direction = 0;
    void* m_in;
    void* m_out;
};

std::string cufftErrorToString (const cufftResult& err) {
    const auto res2string = std::map<cufftResult, std::string>{
        {CUFFT_SUCCESS, "CUFFT_SUCCESS"},
        {CUFFT_INVALID_PLAN,"CUFFT_INVALID_PLAN"},
        {CUFFT_ALLOC_FAILED,"CUFFT_ALLOC_FAILED"},
        {CUFFT_INVALID_TYPE,"CUFFT_INVALID_TYPE"},
        {CUFFT_INVALID_VALUE,"CUFFT_INVALID_VALUE"},
        {CUFFT_INTERNAL_ERROR,"CUFFT_INTERNAL_ERROR"},
        {CUFFT_EXEC_FAILED,"CUFFT_EXEC_FAILED"},
        {CUFFT_SETUP_FAILED,"CUFFT_SETUP_FAILED"},
        {CUFFT_INVALID_SIZE,"CUFFT_INVALID_SIZE"},
        {CUFFT_UNALIGNED_DATA,"CUFFT_UNALIGNED_DATA"}
    };

    const auto it = res2string.find(err);
    if(it != res2string.end()){
        return it->second;
    } else {
        return std::to_string(err) + " (unknown error code)";
    }
}

void assert_cufft_status (std::string const& name, const cufftResult& status) {
    if (status != CUFFT_SUCCESS) {
        amrex::Abort(name + " failed! Error: " + cufftErrorToString(status));
    }
}

std::size_t AnyFFT::Initialize (FFTType type, int nx, int ny) {
    // https://docs.nvidia.com/cuda/cufft/index.html#cufft-api-reference
    m_plan = new VendorPlan;

    cufftType transform_type;
    int rank = 0;
    long long int n[2] = {0, 0};
    long long int batch = 0;

    switch (type) {
        case FFTType::C2C_2D_fwd:
            transform_type = use_float ? CUFFT_C2C : CUFFT_Z2Z;
            m_plan->m_direction = CUFFT_FORWARD;
            rank = 2;
            n[0] = ny;
            n[1] = nx;
            batch = 1;
            break;
        case FFTType::C2C_2D_bkw:
            transform_type = use_float ? CUFFT_C2C : CUFFT_Z2Z;
            m_plan->m_direction = CUFFT_INVERSE;
            rank = 2;
            n[0] = ny;
            n[1] = nx;
            batch = 1;
            break;
        case FFTType::C2R_2D:
            transform_type = use_float ? CUFFT_C2R : CUFFT_Z2D;
            rank = 2;
            n[0] = ny;
            n[1] = nx;
            batch = 1;
            break;
        case FFTType::R2C_2D:
            transform_type = use_float ? CUFFT_R2C : CUFFT_D2Z;
            rank = 2;
            n[0] = ny;
            n[1] = nx;
            batch = 1;
            break;
        case FFTType::R2R_2D:
            amrex::Abort("R2R FFT not supported by cufft");
            return 0;
        case FFTType::C2R_1D_batched:
            transform_type = use_float ? CUFFT_C2R : CUFFT_Z2D;
            rank = 1;
            n[0] = nx;
            batch = ny;
            break;
        case FFTType::R2C_1D_batched:
            transform_type = use_float ? CUFFT_R2C : CUFFT_D2Z;
            rank = 1;
            n[0] = nx;
            batch = ny;
            break;
    }

    cufftResult result;

    result = cufftCreate(&(m_plan->m_cufftplan));
    assert_cufft_status("cufftCreate", result);

    result = cufftSetAutoAllocation(m_plan->m_cufftplan, 0);
    assert_cufft_status("cufftSetAutoAllocation", result);

    std::size_t workSize = 0;

    result = cufftMakePlanMany64(
        m_plan->m_cufftplan,
        rank,
        n,
        nullptr, 0, 0,
        nullptr, 0, 0,
        transform_type,
        batch,
        &workSize);
    assert_cufft_status("cufftMakePlanMany64", result);

    result = cufftSetStream(m_plan->m_cufftplan, amrex::Gpu::Device::cudaStream());
    assert_cufft_status("cufftSetStream", result);

    return workSize;
}

void AnyFFT::SetBuffers (void* in, void* out, void* work_area) {
    m_plan->m_in = in;
    m_plan->m_out = out;

    cufftResult result;

    result = cufftSetWorkArea(m_plan->m_cufftplan, work_area);
    assert_cufft_status("cufftSetWorkArea", result);
}

void AnyFFT::Execute () {
    cufftResult result;

    result = cufftXtExec(m_plan->m_cufftplan, m_plan->m_in, m_plan->m_out, m_plan->m_direction);
    assert_cufft_status("cufftXtExec", result);
}

AnyFFT::~AnyFFT () {
    if (m_plan) {
        cufftResult result;

        result = cufftDestroy(m_plan->m_cufftplan);
        assert_cufft_status("cufftDestroy", result);

        delete m_plan;
    }
}
