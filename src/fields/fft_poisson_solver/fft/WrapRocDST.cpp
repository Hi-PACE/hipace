/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "AnyDST.H"
#include "RocFFTUtils.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Parser.H"

#include <AMReX_Config.H>

namespace AnyDST
{
    // see WrapCuDST for documentation
    void ExpandR2R (amrex::FArrayBox& dst, amrex::FArrayBox& src)
    {
        constexpr int scomp = 0;
        constexpr int dcomp = 0;

        const amrex::Box bx = src.box();
        const int nx = bx.length(0);
        const int ny = bx.length(1);
        const amrex::IntVect lo = bx.smallEnd();
        Array2<amrex::Real const> const src_array = src.const_array(scomp);
        Array2<amrex::Real> const dst_array = dst.array(dcomp);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int)
            {
                /* upper left quadrant */
                dst_array(i+1,j+1) = src_array(i, j);
                /* lower left quadrant */
                dst_array(i+1,j+ny+2) = -src_array(i, ny-1-j+2*lo[1]);
                /* upper right quadrant */
                dst_array(i+nx+2,j+1) = -src_array(nx-1-i+2*lo[0], j);
                /* lower right quadrant */
                dst_array(i+nx+2,j+ny+2) = src_array(nx-1-i+2*lo[0], ny-1-j+2*lo[1]);
            }
            );
    }

    // see WrapCuDST for documentation
    void ShrinkC2R (amrex::FArrayBox& dst, amrex::BaseFab<amrex::GpuComplex<amrex::Real>>& src)
    {
        constexpr int scomp = 0;
        constexpr int dcomp = 0;

        const amrex::Box bx = dst.box();
        Array2<amrex::GpuComplex<amrex::Real> const> const src_array = src.const_array(scomp);
        Array2<amrex::Real> const dst_array = dst.array(dcomp);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int)
            {
                /* upper left quadrant */
                dst_array(i,j) = -src_array(i+1, j+1).real();
            }
            );
    }

    // see WrapCuDST for documentation
    void ToComplex (const amrex::Real* const AMREX_RESTRICT in,
                    amrex::GpuComplex<amrex::Real>* const AMREX_RESTRICT out,
                    const int n_data, const int n_batch)
    {
        const int n_half = (n_data+1)/2;
        if((n_data%2 == 1)) {
            amrex::ParallelFor({{0,0,0}, {n_half,n_batch-1,0}},
                [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
                {
                    const int stride_in = n_data*j;
                    const int i_is_zero = (i==0);
                    const int i_is_n_half = (i==n_half);
                    const amrex::Real real = -in[2*i-2+2*i_is_zero+stride_in]*(1-2*i_is_zero)
                                             +in[2*i-2*i_is_n_half+stride_in]*(1-2*i_is_n_half);
                    const amrex::Real imag = in[2*i-1+i_is_zero-i_is_n_half+stride_in]
                                             *!i_is_zero*!i_is_n_half;
                    out[i+(n_half+1)*j] = amrex::GpuComplex<amrex::Real>(real, imag);
                });
        } else {
            amrex::ParallelFor({{0,0,0}, {n_half,n_batch-1,0}},
                [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
                {
                    const int stride_in = n_data*j;
                    const int i_is_zero = (i==0);
                    const int i_is_n_half = (i==n_half);
                    const amrex::Real real = -in[2*i-2+2*i_is_zero+stride_in]*(1-2*i_is_zero)
                                             +in[2*i-i_is_n_half+stride_in]*!i_is_n_half;
                    const amrex::Real imag = in[2*i-1+i_is_zero+stride_in]*!i_is_zero;
                    out[i+(n_half+1)*j] = amrex::GpuComplex<amrex::Real>(real, imag);
                });
        }
    }

    // see WrapCuDST for documentation
    void C2Rfft (AnyFFT::VendorFFTPlan& plan, amrex::GpuComplex<amrex::Real>* AMREX_RESTRICT in,
                 amrex::Real* const AMREX_RESTRICT out, rocfft_execution_info execinfo)
    {
        rocfft_status result;

        void* in_arr[2] = {in, nullptr};
        void* out_arr[2] = {out, nullptr};
        result = rocfft_execute(plan, in_arr, out_arr, execinfo);

        RocFFTUtils::assert_rocfft_status("rocfft_execute", result);
    }

    // see WrapCuDST for documentation
    void ToSine (const amrex::Real* const AMREX_RESTRICT in, amrex::Real* const AMREX_RESTRICT out,
                 const int n_data, const int n_batch)
    {
        using namespace amrex::literals;

        const amrex::Real n_1_real = n_data+1._rt;
        const int n_1 = n_data+1;
        amrex::ParallelFor({{1,0,0}, {(n_data+1)/2,n_batch-1,0}},
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
            {
                const int i_rev = n_1-i;
                const int stride_in = n_1*j;
                const int stride_out = n_data*j;
                const amrex::Real in_a = in[i+stride_in];
                const amrex::Real in_b = in[i_rev+stride_in];
#ifdef AMREX_USE_FLOAT
                out[i-1+stride_out] = 0.5_rt*(in_b-in_a+(in_a+in_b)/(2._rt*sinpif(i/n_1_real)));
                out[i_rev-1+stride_out] = 0.5_rt*(in_a-in_b+(in_a+in_b)/(2._rt
                                          *sinpif(i_rev/n_1_real)));
#else
                out[i-1+stride_out] = 0.5_rt*(in_b-in_a+(in_a+in_b)/(2._rt*sinpi(i/n_1_real)));
                out[i_rev-1+stride_out] = 0.5_rt*(in_a-in_b+(in_a+in_b)/(2._rt
                                          *sinpi(i_rev/n_1_real)));
#endif
            });
    }

    // see WrapCuDST for documentation
    void Transpose (const amrex::Real* const AMREX_RESTRICT in,
                    amrex::Real* const AMREX_RESTRICT out,
                    const int n_data, const int n_batch)
    {
        constexpr int tile_dim = 32; //must be power of 2
        constexpr int block_rows = 8;
        const int num_blocks_x = (n_data + tile_dim - 1)/tile_dim;
        const int num_blocks_y = (n_batch + tile_dim - 1)/tile_dim;
        amrex::launch(num_blocks_x*num_blocks_y, tile_dim*block_rows,
            tile_dim*(tile_dim+1)*sizeof(amrex::Real), amrex::Gpu::gpuStream(),
            [=] AMREX_GPU_DEVICE() noexcept
            {
                amrex::Gpu::SharedMemory<amrex::Real> gsm;
                amrex::Real* const tile = gsm.dataPtr();

                const int thread_x = threadIdx.x&(tile_dim-1);
                const int thread_y = threadIdx.x/tile_dim;
                const int block_y = blockIdx.x/num_blocks_x;
                const int block_x = blockIdx.x - block_y*num_blocks_x;
                int mat_x = block_x * tile_dim + thread_x;
                int mat_y = block_y * tile_dim + thread_y;

                for (int i = 0; i < tile_dim; i += block_rows) {
                    if(mat_x < n_data && (mat_y+i) < n_batch) {
                        tile[(thread_y+i)*(tile_dim+1) + thread_x] = in[(mat_y+i)*n_data + mat_x];
                    }
                }

                __syncthreads();

                mat_x = block_y * tile_dim + thread_x;
                mat_y = block_x * tile_dim + thread_y;

                for (int i = 0; i < tile_dim; i += block_rows) {
                    if(mat_x < n_batch && (mat_y+i) < n_data) {
                        out[(mat_y+i)*n_batch + mat_x] = tile[thread_x*(tile_dim+1) + thread_y+i];
                    }
                }
            });
    }

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array)
    {
        HIPACE_PROFILE("AnyDST::CreatePlan()");
        DSTplan dst_plan;

        amrex::ParmParse pp("hipace");
        dst_plan.use_small_dst = (std::max(real_size[0], real_size[1]) >= 511);
        queryWithParser(pp, "use_small_dst", dst_plan.use_small_dst);

        if(!dst_plan.use_small_dst) {
            const int nx = real_size[0];
            const int ny = real_size[1];
            int dim = 2;

            // Allocate expanded_position_array Real of size (2*nx+2, 2*ny+2)
            // Allocate expanded_fourier_array Complex of size (nx+2, 2*ny+2)
            amrex::Box expanded_position_box {{0, 0, 0}, {2*nx+1, 2*ny+1, 0}};
            amrex::Box expanded_fourier_box {{0, 0, 0}, {nx+1, 2*ny+1, 0}};
            // shift box to match rest of fields
            expanded_position_box += position_array->box().smallEnd();
            expanded_fourier_box += fourier_array->box().smallEnd();
            dst_plan.m_expanded_position_array =
                std::make_unique<amrex::FArrayBox>(
                    expanded_position_box, 1);
            dst_plan.m_expanded_fourier_array =
                std::make_unique<amrex::BaseFab<amrex::GpuComplex<amrex::Real>>>(
                    expanded_fourier_box, 1);

            // setting the initial values to 0
            // we don't set the expanded Fourier array, because it will be initialized by the FFT
            dst_plan.m_expanded_position_array->setVal<amrex::RunOn::Device>(0.,
                dst_plan.m_expanded_position_array->box(), 0,
                dst_plan.m_expanded_position_array->nComp());

            // check for type of expanded size, should be const size_t *
            const amrex::IntVect& expanded_size = expanded_position_box.length();
            const std::size_t lengths[] = {AMREX_D_DECL(std::size_t(expanded_size[0]),
                                                        std::size_t(expanded_size[1]),
                                                        std::size_t(expanded_size[2]))};

    #ifdef AMREX_USE_FLOAT
            rocfft_precision precision = rocfft_precision_single;
    #else
            rocfft_precision precision = rocfft_precision_double;
    #endif

            // Initialize fft_plan.m_plan with the vendor fft plan.
            rocfft_status result;
            result = rocfft_plan_create(&(dst_plan.m_plan),
                                        rocfft_placement_notinplace,
                                        rocfft_transform_type_real_forward,
                                        precision,
                                        dim,
                                        lengths,
                                        1,
                                        nullptr);

            RocFFTUtils::assert_rocfft_status("rocfft_plan_create", result);

            std::size_t buffersize = 0;
            result = rocfft_plan_get_work_buffer_size(dst_plan.m_plan, &buffersize);
            RocFFTUtils::assert_rocfft_status("rocfft_plan_get_work_buffer_size", result);

            result = rocfft_execution_info_create(&(dst_plan.m_execinfo));
            RocFFTUtils::assert_rocfft_status("rocfft_execution_info_create", result);

            dst_plan.m_buffer = amrex::The_Arena()->alloc(buffersize);
            result = rocfft_execution_info_set_work_buffer(dst_plan.m_execinfo, dst_plan.m_buffer,
                                                           buffersize);
            RocFFTUtils::assert_rocfft_status("rocfft_execution_info_set_work_buffer", result);

            result = rocfft_execution_info_set_stream(dst_plan.m_execinfo, amrex::Gpu::gpuStream());
            RocFFTUtils::assert_rocfft_status("rocfft_execution_info_set_stream", result);

            std::size_t mb = 1024*1024;

            amrex::Print() << "using R2C rocFFT of size " << expanded_size[0] << " * "
                << expanded_size[1] << " with " << (buffersize+mb-1)/mb << " MiB of work area\n";

            // Store meta-data in dst_plan
            dst_plan.m_position_array = position_array;
            dst_plan.m_fourier_array = fourier_array;

            return dst_plan;
        }
        else {
            const int nx = real_size[0]; // contiguous
            const int ny = real_size[1]; // not contiguous

            // Allocate 1d Array for 2d data or 2d transpose data
            const int real_1d_size = std::max((nx+1)*ny, (ny+1)*nx);
            const int complex_1d_size = std::max(((nx+1)/2+1)*ny, ((ny+1)/2+1)*nx);
            amrex::Box real_box {{0, 0, 0}, {real_1d_size-1, 0, 0}};
            amrex::Box complex_box {{0, 0, 0}, {complex_1d_size-1, 0, 0}};
            dst_plan.m_expanded_position_array =
                std::make_unique<amrex::FArrayBox>(
                    real_box, 1);
            dst_plan.m_expanded_fourier_array =
                std::make_unique<amrex::BaseFab<amrex::GpuComplex<amrex::Real>>>(
                    complex_box, 1);

#ifdef AMREX_USE_FLOAT
            rocfft_precision precision = rocfft_precision_single;
#else
            rocfft_precision precision = rocfft_precision_double;
#endif

            rocfft_status result;

            const std::size_t s_1[3] = {std::size_t(nx+1) ,0u ,0u};

            result = rocfft_plan_create(&(dst_plan.m_plan),
                                        rocfft_placement_notinplace,
                                        rocfft_transform_type_real_inverse,
                                        precision,
                                        1,
                                        s_1,
                                        ny,
                                        nullptr);

            RocFFTUtils::assert_rocfft_status("rocfft_plan_create", result);

            const std::size_t s_2[3] = {std::size_t(ny+1) ,0u ,0u};

            result = rocfft_plan_create(&(dst_plan.m_plan_b),
                                        rocfft_placement_notinplace,
                                        rocfft_transform_type_real_inverse,
                                        precision,
                                        1,
                                        s_2,
                                        nx,
                                        nullptr);

            RocFFTUtils::assert_rocfft_status("rocfft_plan_create", result);

            std::size_t work_size = 0;
            std::size_t work_size_b = 0;

            result = rocfft_plan_get_work_buffer_size(dst_plan.m_plan, &work_size);
            RocFFTUtils::assert_rocfft_status("rocfft_plan_get_work_buffer_size", result);

            result = rocfft_plan_get_work_buffer_size(dst_plan.m_plan_b, &work_size_b);
            RocFFTUtils::assert_rocfft_status("rocfft_plan_get_work_buffer_size", result);

            result = rocfft_execution_info_create(&(dst_plan.m_execinfo));
            RocFFTUtils::assert_rocfft_status("rocfft_execution_info_create", result);

            std::size_t buffersize = std::max(work_size, work_size_b);
            dst_plan.m_buffer = amrex::The_Arena()->alloc(buffersize);

            result = rocfft_execution_info_set_work_buffer(dst_plan.m_execinfo, dst_plan.m_buffer,
                                                           buffersize);
            RocFFTUtils::assert_rocfft_status("rocfft_execution_info_set_work_buffer", result);

            result = rocfft_execution_info_set_stream(dst_plan.m_execinfo, amrex::Gpu::gpuStream());
            RocFFTUtils::assert_rocfft_status("rocfft_execution_info_set_stream", result);

            std::size_t mb = 1024*1024;

            amrex::Print() << "using C2R rocFFT of sizes " << s_1[0] << " and "
                << s_2[0] << " with " << (buffersize+mb-1)/mb << " MiB of work area\n";

            // Store meta-data in dst_plan
            dst_plan.m_position_array = position_array;
            dst_plan.m_fourier_array = fourier_array;

            return dst_plan;
        }
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
        rocfft_plan_destroy( dst_plan.m_plan );
        rocfft_plan_destroy( dst_plan.m_plan_b );

        amrex::The_Arena()->free(dst_plan.m_buffer);
        rocfft_execution_info_destroy(dst_plan.m_execinfo);
    }

    void Execute (DSTplan& dst_plan, direction d){
        HIPACE_PROFILE("AnyDST::Execute()");

        if(!dst_plan.use_small_dst) {
            // Swap position and fourier space based on execute direction
            amrex::FArrayBox* position_array =
                (d == direction::forward) ? dst_plan.m_position_array : dst_plan.m_fourier_array;
            amrex::FArrayBox* fourier_array =
                (d == direction::forward) ? dst_plan.m_fourier_array : dst_plan.m_position_array;

            // Expand in position space m_position_array -> m_expanded_position_array
            ExpandR2R(*dst_plan.m_expanded_position_array, *position_array);

            rocfft_status result;

            // R2C FFT m_expanded_position_array -> m_expanded_fourier_array
            void* in[2] = {dst_plan.m_expanded_position_array->dataPtr(), nullptr};
            void* out[2] = {dst_plan.m_expanded_fourier_array->dataPtr(), nullptr};
            result = rocfft_execute(dst_plan.m_plan, in, out, dst_plan.m_execinfo);

            RocFFTUtils::assert_rocfft_status("rocfft_execute", result);

            // Shrink in Fourier space m_expanded_fourier_array -> m_fourier_array
            ShrinkC2R(*fourier_array, *dst_plan.m_expanded_fourier_array);
        }
        else {
            const int nx = dst_plan.m_position_array->box().length(0); // initially contiguous
            const int ny = dst_plan.m_position_array->box().length(1); // contiguous after transpose

            amrex::Real* const tmp_pos_arr = dst_plan.m_position_array->dataPtr();
            amrex::Real* const tmp_fourier_arr = dst_plan.m_fourier_array->dataPtr();
            amrex::GpuComplex<amrex::Real>* comp_arr = dst_plan.m_expanded_fourier_array->dataPtr();
            amrex::Real* const real_arr = dst_plan.m_expanded_position_array->dataPtr();

            // Swap position and fourier space based on execute direction
            amrex::Real* const pos_arr =
                (d == direction::forward) ? tmp_pos_arr : tmp_fourier_arr;
            amrex::Real* const fourier_arr =
                (d == direction::forward) ? tmp_fourier_arr : tmp_pos_arr;

            ToComplex(pos_arr, comp_arr, nx, ny);

            C2Rfft(dst_plan.m_plan, comp_arr, real_arr, dst_plan.m_execinfo);

            ToSine(real_arr, pos_arr, nx, ny);

            Transpose(pos_arr, fourier_arr, nx, ny);

            ToComplex(fourier_arr, comp_arr, ny, nx);

            C2Rfft(dst_plan.m_plan_b, comp_arr, real_arr, dst_plan.m_execinfo);

            ToSine(real_arr, pos_arr, ny, nx);

            Transpose(pos_arr, fourier_arr, ny, nx);
        }
    }
}
