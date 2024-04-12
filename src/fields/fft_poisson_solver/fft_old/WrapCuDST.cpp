/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "AnyDST.H"
#include "CuFFTUtils.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Parser.H"

#include <AMReX_Config.H>

namespace AnyDST
{
#ifdef AMREX_USE_FLOAT
    cufftType VendorR2C = CUFFT_R2C;
    cufftType VendorC2R = CUFFT_C2R;
#else
    cufftType VendorR2C = CUFFT_D2Z;
    cufftType VendorC2R = CUFFT_Z2D;
#endif

    /** \brief Extend src into a symmetrized larger array dst
     *
     * \param[in,out] dst destination array, odd symmetry around 0 and the middle points in x and y
     * \param[in] src source array
     */
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
    };

    /** \brief Extract symmetrical src array into smaller array dst
     *
     * \param[in,out] dst destination array
     * \param[in] src destination array, symmetric in x and y
     */
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
    };

    /** \brief Make Complex array out of Real array to prepare for fft.
     * out[idx] = -in[2*idx-2] + in[2*idx] + i*in[2*idx-1] for each column with
     * in[-1] = 0; in[-2] = -in[0]; in[n_data] = 0; in[n_data+1] = -in[n_data-1]
     *
     * \param[in] in input real array
     * \param[out] out output complex array
     * \param[in] n_data number of (contiguous) rows in position matrix
     * \param[in] n_batch number of (strided) columns in position matrix
     */
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
    };

    /** \brief Complex to Real fft for every column of the input matrix.
     * The output Matrix has its indexes reversed compared to some other libraries
     *
     * \param[in] plan cuda fft plan for transformation
     * \param[in] in input complex array
     * \param[out] out output real array
     */
    void C2Rfft (AnyFFT::VendorFFTPlan& plan, amrex::GpuComplex<amrex::Real>* AMREX_RESTRICT in,
                 amrex::Real* const AMREX_RESTRICT out)
    {
        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
        cufftSetStream(plan, stream);
        cufftResult result;

#ifdef AMREX_USE_FLOAT
        result = cufftExecC2R(plan, reinterpret_cast<AnyFFT::Complex*>(in), out);
#else
        result = cufftExecZ2D(plan, reinterpret_cast<AnyFFT::Complex*>(in), out);
#endif
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " forward transform using cufftExec failed ! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
    };

    /** \brief Make Sine-space Real array out of array from fft.
     * out[idx] = 0.5 *(in[n_data-idx] - in[idx+1] + (in[n_data-idx] + in[idx+1])/
     * (2*sin((idx+1)*pi/(n_data+1)))) for each column
     *
     * \param[in] in input real array
     * \param[out] out output real array
     * \param[in] n_data number of (contiguous) rows in position matrix
     * \param[in] n_batch number of (strided) columns in position matrix
     */
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
    };

    /** \brief Transpose input matrix
     * out[idy][idx] = in[idx][idy]
     *
     * \param[in] in input real array
     * \param[out] out output real array
     * \param[in] n_data number of (contiguous) rows in input matrix
     * \param[in] n_batch number of (strided) columns in input matrix
     */
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
    };

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array)
    {
        HIPACE_PROFILE("AnyDST::CreatePlan()");
        DSTplan dst_plan;

        amrex::ParmParse pp("hipace");
        dst_plan.use_small_dst = (std::max(real_size[0], real_size[1]) >= 511);
        queryWithParser(pp, "use_small_dst", dst_plan.use_small_dst);

        if (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 1) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE((std::max(real_size[0], real_size[1]) <= 1024),
            "Due to a bug in cuFFT, CUDA 11.1 supports only nx, ny <= 1024. Please use CUDA "
            "version >= 11.2 (recommended) or <= 11.0 for larger grid sizes.");
        }

        if(!dst_plan.use_small_dst) {
            const int nx = real_size[0];
            const int ny = real_size[1];

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

            const amrex::IntVect& expanded_size = expanded_position_box.length();

            // Initialize fft_plan.m_plan with the vendor fft plan.
            cufftResult result;
            result = cufftPlan2d(
                &(dst_plan.m_plan), expanded_size[1], expanded_size[0], VendorR2C);

            if ( result != CUFFT_SUCCESS ) {
                amrex::Print() << " cufftplan failed! Error: " <<
                    CuFFTUtils::cufftErrorToString(result) << "\n";
            }

            std::size_t buffersize = 0;
            cufftGetSize(dst_plan.m_plan, &buffersize);

            std::size_t mb = 1024*1024;

            amrex::Print() << "using R2C cuFFT of size " << expanded_size[0] << " * "
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

            // Initialize fft_plan.m_plan with the vendor fft plan.
            int s_1 = nx+1;
            cufftResult result;
            result = cufftPlanMany(
                &(dst_plan.m_plan), 1, &s_1, NULL, 1, (nx+1)/2+1, NULL, 1, nx+1, VendorC2R, ny);

            if ( result != CUFFT_SUCCESS ) {
                amrex::Print() << " cufftplan failed! Error: " <<
                    CuFFTUtils::cufftErrorToString(result) << "\n";
            }

            // Initialize transposed fft_plan.m_plan_b with the vendor fft plan.
            int s_2 = ny+1;
            cufftResult resultb;
            resultb = cufftPlanMany(
                &(dst_plan.m_plan_b), 1, &s_2, NULL, 1, (ny+1)/2+1, NULL, 1, ny+1, VendorC2R, nx);

            if ( resultb != CUFFT_SUCCESS ) {
                amrex::Print() << " cufftplan failed! Error: " <<
                    CuFFTUtils::cufftErrorToString(resultb) << "\n";
            }

            std::size_t buffersize = 0;
            std::size_t buffersize_b = 0;
            cufftGetSize(dst_plan.m_plan, &buffersize);
            cufftGetSize(dst_plan.m_plan_b, &buffersize_b);

            std::size_t mb = 1024*1024;

            amrex::Print() << "using C2R cuFFT of sizes " << s_1 << " and "
                << s_2 << " with " << (buffersize+buffersize_b+mb-1)/mb << " MiB of work area\n";

            // Store meta-data in dst_plan
            dst_plan.m_position_array = position_array;
            dst_plan.m_fourier_array = fourier_array;

            return dst_plan;
        }
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
        cufftDestroy( dst_plan.m_plan );
        cufftDestroy( dst_plan.m_plan_b );
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

            cudaStream_t stream = amrex::Gpu::Device::cudaStream();
            cufftSetStream ( dst_plan.m_plan, stream);
            cufftResult result;

            // R2C FFT m_expanded_position_array -> m_expanded_fourier_array
#ifdef AMREX_USE_FLOAT
            result = cufftExecR2C(
                dst_plan.m_plan, dst_plan.m_expanded_position_array->dataPtr(),
                reinterpret_cast<AnyFFT::Complex*>(dst_plan.m_expanded_fourier_array->dataPtr()));
#else
            result = cufftExecD2Z(
                dst_plan.m_plan, dst_plan.m_expanded_position_array->dataPtr(),
                reinterpret_cast<AnyFFT::Complex*>(dst_plan.m_expanded_fourier_array->dataPtr()));
#endif
            // Shrink in Fourier space m_expanded_fourier_array -> m_fourier_array
            ShrinkC2R(*fourier_array, *dst_plan.m_expanded_fourier_array);

            if ( result != CUFFT_SUCCESS ) {
                amrex::Print() << " forward transform using cufftExec failed ! Error: " <<
                    CuFFTUtils::cufftErrorToString(result) << "\n";
            }
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

            C2Rfft(dst_plan.m_plan, comp_arr, real_arr);

            ToSine(real_arr, pos_arr, nx, ny);

            Transpose(pos_arr, fourier_arr, nx, ny);

            ToComplex(fourier_arr, comp_arr, ny, nx);

            C2Rfft(dst_plan.m_plan_b, comp_arr, real_arr);

            ToSine(real_arr, pos_arr, ny, nx);

            Transpose(pos_arr, fourier_arr, ny, nx);
        }
    }
}
