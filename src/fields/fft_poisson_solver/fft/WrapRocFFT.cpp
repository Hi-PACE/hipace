/* Copyright 2021-2024
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "AnyFFT.H"

#include <AMReX.H>
#include <AMReX_GpuDevice.H>

#if __has_include(<rocfft/rocfft.h>)  // ROCm 5.3+
#include <rocfft/rocfft.h>
#else
#include <rocfft.h>
#endif

#include <string>

#ifdef AMREX_USE_FLOAT
static constexpr bool use_float = true;
#else
static constexpr bool use_float = false;
#endif

struct VendorPlan {
    rocfft_plan m_rocfftplan;
    rocfft_execution_info m_execinfo;
    std::size_t m_work_size = 0;
    void* m_in;
    void* m_out;
};

std::string rocfftErrorToString (const rocfft_status& err) {
    if              (err == rocfft_status_success) {
        return std::string("rocfft_status_success");
    } else if       (err == rocfft_status_failure) {
        return std::string("rocfft_status_failure");
    } else if       (err == rocfft_status_invalid_arg_value) {
        return std::string("rocfft_status_invalid_arg_value");
    } else if       (err == rocfft_status_invalid_dimensions) {
        return std::string("rocfft_status_invalid_dimensions");
    } else if       (err == rocfft_status_invalid_array_type) {
        return std::string("rocfft_status_invalid_array_type");
    } else if       (err == rocfft_status_invalid_strides) {
        return std::string("rocfft_status_invalid_strides");
    } else if       (err == rocfft_status_invalid_distance) {
        return std::string("rocfft_status_invalid_distance");
    } else if       (err == rocfft_status_invalid_offset) {
        return std::string("rocfft_status_invalid_offset");
    } else if       (err == rocfft_status_invalid_work_buffer) {
        return std::string("rocfft_status_invalid_work_buffer");
    }else {
        return std::to_string(err) + " (unknown error code)";
    }
}

void assert_rocfft_status (std::string const& name, const rocfft_status& status) {
    if (status != rocfft_status_success) {
        amrex::Abort(name + " failed! Error: " + rocfftErrorToString(status));
    }
}

std::size_t AnyFFT::Initialize (FFTType type, int nx, int ny) {
    // https://rocm.docs.amd.com/projects/rocFFT/en/latest/reference/allapi.html#
    m_plan = new VendorPlan;

    rocfft_transform_type transform_type;
    std::size_t dimensions = 0;
    // lengths is in FORTRAN order
    std::size_t lengths[2] = {0, 0};
    std::size_t number_of_transforms = 0;

    switch (type) {
        case FFTType::C2C_2D_fwd:
            transform_type = rocfft_transform_type_complex_forward;
            dimensions = 2;
            lengths[0] = nx;
            lengths[1] = ny;
            number_of_transforms = 1;
            break;
        case FFTType::C2C_2D_bkw:
            transform_type = rocfft_transform_type_complex_inverse;
            dimensions = 2;
            lengths[0] = nx;
            lengths[1] = ny;
            number_of_transforms = 1;
            break;
        case FFTType::C2R_2D:
            transform_type = rocfft_transform_type_real_inverse;
            dimensions = 2;
            lengths[0] = nx;
            lengths[1] = ny;
            number_of_transforms = 1;
            break;
        case FFTType::R2C_2D:
            transform_type = rocfft_transform_type_real_forward;
            dimensions = 2;
            lengths[0] = nx;
            lengths[1] = ny;
            number_of_transforms = 1;
            break;
        case FFTType::R2R_2D:
            amrex::Abort("R2R FFT not supported by rocfft");
            return 0;
        case FFTType::C2R_1D_batched:
            transform_type = rocfft_transform_type_real_inverse;
            dimensions = 1;
            lengths[0] = nx;
            number_of_transforms = ny;
            break;
    }

    rocfft_status status;

    status = rocfft_plan_create(
        &(m_plan->m_rocfftplan),
        rocfft_placement_notinplace,
        transform_type,
        use_float ? rocfft_precision_single : rocfft_precision_double,
        dimensions,
        lengths,
        number_of_transforms,
        nullptr);
    assert_rocfft_status("rocfft_plan_create", status);

    status = rocfft_plan_get_work_buffer_size(m_plan->m_rocfftplan, &(m_plan->m_work_size));
    assert_rocfft_status("rocfft_plan_get_work_buffer_size", status);

    return m_plan->m_work_size;
}

void AnyFFT::SetBuffers (void* in, void* out, void* work_area) {
    m_plan->m_in = in;
    m_plan->m_out = out;

    rocfft_status status;

    status = rocfft_execution_info_create(&(m_plan->m_execinfo));
    assert_rocfft_status("rocfft_execution_info_create", status);

    if (m_plan->m_work_size > 0) {
        status = rocfft_execution_info_set_work_buffer(
            m_plan->m_execinfo, work_area, m_plan->m_work_size);
        assert_rocfft_status("rocfft_execution_info_set_work_buffer", status);
    }

    status = rocfft_execution_info_set_stream(m_plan->m_execinfo, amrex::Gpu::gpuStream());
    assert_rocfft_status("rocfft_execution_info_set_stream", status);
}

void AnyFFT::Execute () {
    rocfft_status status;

    void* in_arr[2] = {m_plan->m_in, nullptr};
    void* out_arr[2] = {m_plan->m_out, nullptr};

    status = rocfft_execute(m_plan->m_rocfftplan, in_arr, out_arr, m_plan->m_execinfo);
    assert_rocfft_status("rocfft_execute", status);
}

AnyFFT::~AnyFFT () {
    if (m_plan) {
        rocfft_status status;

        status = rocfft_execution_info_destroy(m_plan->m_execinfo);
        assert_rocfft_status("rocfft_execution_info_destroy", status);

        status = rocfft_plan_destroy(m_plan->m_rocfftplan);
        assert_rocfft_status("rocfft_plan_destroy", status);

        delete m_plan;
    }
}

void AnyFFT::setup () {
    rocfft_status status;

    status = rocfft_setup();
    assert_rocfft_status("rocfft_setup", status);
}

void AnyFFT::cleanup () {
    rocfft_status status;

    status = rocfft_cleanup();
    assert_rocfft_status("rocfft_cleanup", status);
}
