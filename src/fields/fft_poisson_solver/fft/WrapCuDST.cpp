#include "AnyDST.H"
#include "CuFFTUtils.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include <cmath>

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
        HIPACE_PROFILE("AnyDST::ExpandR2R()");
        constexpr int scomp = 0;
        constexpr int dcomp = 0;

        const amrex::Box bx = src.box();
        const int nx = bx.length(0);
        const int ny = bx.length(1);
        amrex::Array4<amrex::Real const> const & src_array = src.array();
        amrex::Array4<amrex::Real> const & dst_array = dst.array();

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* upper left quadrant */
                dst_array(i+1,j+1,0,dcomp) = src_array(i, j, k, scomp);
                /* lower left quadrant */
                dst_array(i+1,j+ny+2,0,dcomp) = -src_array(i, ny-1-j, k, scomp);
                /* upper right quadrant */
                dst_array(i+nx+2,j+1,0,dcomp) = -src_array(nx-1-i, j, k, scomp);
                /* lower right quadrant */
                dst_array(i+nx+2,j+ny+2,0,dcomp) = src_array(nx-1-i, ny-1-j, k, scomp);
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
        HIPACE_PROFILE("AnyDST::ShrinkC2R()");
        constexpr int scomp = 0;
        constexpr int dcomp = 0;

        const amrex::Box bx = dst.box();
        amrex::Array4<amrex::GpuComplex<amrex::Real> const> const & src_array = src.array();
        amrex::Array4<amrex::Real> const & dst_array = dst.array();
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                /* upper left quadrant */
                dst_array(i,j,k,dcomp) = -src_array(i+1, j+1, 0, scomp).real();
            }
            );
    };

    void ToComplex(const amrex::Real* const in, amrex::GpuComplex<amrex::Real>* const out,
                   const int n_data, const int n_batch)
    {
        {
        HIPACE_PROFILE("AnyDST::ToComplex()");
        const int n_half = (n_data+1)/2;
        if((n_data%2 == 1)) {
            amrex::ParallelFor({{0,0,0}, {n_half,n_batch-1,0}},
                [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
                {
                    const int stride_in = n_data*j;
                    const int stride_out = (n_half+1)*j;
                    if(i==0) {
                        out[stride_out].m_real = 2*in[stride_in];
                        out[stride_out].m_imag = 0;
                    } else if(i==n_half) {
                        out[n_half+stride_out].m_real = -2*in[n_data-1+stride_in];
                        out[n_half+stride_out].m_imag = 0;
                    } else {
                        out[i+stride_out].m_real = - in[2*i-2+stride_in] +in[2*i+stride_in];
                        out[i+stride_out].m_imag = in[2*i-1+stride_in];
                    }
                });
        } else {
            amrex::ParallelFor({{0,0,0}, {n_half,n_batch-1,0}},
                [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
                {
                    const int stride_in = n_data*j;
                    const int stride_out = (n_half+1)*j;
                    if(i==0) {
                        out[stride_out].m_real = 2*in[stride_in];
                        out[stride_out].m_imag = 0;
                    } else if(i==n_half) {
                        out[n_half+stride_out].m_real = -in[n_data-2+stride_in];
                        out[n_half+stride_out].m_imag = in[n_data-1+stride_in];
                    } else {
                        out[i+stride_out].m_real = -in[2*i-2+stride_in] +in[2*i+stride_in];
                        out[i+stride_out].m_imag = in[2*i-1+stride_in];
                    }
                });
        }
        amrex::Gpu::synchronize();
        }
    };

    void C2Rfft(AnyFFT::VendorFFTPlan& plan, amrex::GpuComplex<amrex::Real>* in,
                amrex::Real* const out)
    {
        {
        HIPACE_PROFILE("AnyDST::C2Rfft()");
        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
        cufftSetStream(plan, stream);
        cufftResult result;

#ifdef AMREX_USE_FLOAT
        result = cufftExecC2R(plan, reinterpret_cast<AnyFFT::Complex*>(in), out);
#else
        result = cufftExecZ2D(plan, reinterpret_cast<AnyFFT::Complex*>(in), out);
#endif
        cudaDeviceSynchronize();
        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " forward transform using cufftExec failed ! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
        }
    };

    void ToSine(const amrex::Real* const in, amrex::Real* const out,
                const int n_data, const int n_batch)
    {
        {
        HIPACE_PROFILE("AnyDST::ToSine()");
        amrex::Real pi = MathConst::pi;
        amrex::ParallelFor({{1,0,0}, {n_data,n_batch-1,0}},
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
            {
                const int n_1 = n_data+1;
                const int stride_in = n_1*j;
                const int stride_out = n_data*j;
                out[i-1+stride_out] = 0.5*( in[n_1-i+stride_in] - in[i+stride_in] +
                    (in[i+stride_in] + in[n_1-i+stride_in])/(2*std::sin(i*pi/n_1)));
            });
        amrex::Gpu::synchronize();
        }
    };

    void Transpose(const amrex::Real* const in, amrex::Real* const out,
                   const int n_data, const int n_batch)
    {
        {
        HIPACE_PROFILE("AnyDST::TransposeBad()");
        amrex::ParallelFor({{0,0,0}, {n_data-1,n_batch-1,0}},
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
            {
                out[j+n_batch*i] = in[i+n_data*j];
            });
        amrex::Gpu::synchronize();
        }
    };
/*
    void print_arr(const amrex::Real* const arr, const int nx, const int ny) {
        for(int j=0; j<ny;++j) {
            std::cout << "[ ";
            for(int i=0;i<nx;++i) {
                std::cout << arr[i+nx*j] << " , ";
            }
            std::cout << " ], ";
        }
        std::cout << std::endl;
    }

    void print_arr(amrex::GpuComplex<amrex::Real>* arr, const int nx, const int ny) {
        for(int j=0; j<ny;++j) {
            std::cout << "[ ";
            for(int i=0;i<nx;++i) {
                std::cout << arr[i+nx*j].m_real << " +1j* " << arr[i+nx*j].m_imag << " , ";
            }
            std::cout << " ], ";
        }
        std::cout << std::endl;
    }
*/
    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array)
    {
        HIPACE_PROFILE("AnyDST::CreatePlan()");
        DSTplan dst_plan;

        amrex::ParmParse pp("fields");
        dst_plan.use_small_dst = false;
        pp.query("use_small_dst", dst_plan.use_small_dst);

        if(!dst_plan.use_small_dst) {
            const int nx = real_size[0];
            const int ny = real_size[1];

            // Allocate expanded_position_array Real of size (2*nx+2, 2*ny+2)
            // Allocate expanded_fourier_array Complex of size (nx+2, 2*ny+2)
            amrex::Box expanded_position_box {{0, 0, 0}, {2*nx+1, 2*ny+1, 0}};
            amrex::Box expanded_fourier_box {{0, 0, 0}, {nx+1, 2*ny+1, 0}};
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

            // Store meta-data in dst_plan
            dst_plan.m_position_array = position_array;
            dst_plan.m_fourier_array = fourier_array;

            return dst_plan;
        }
        else {
            const int nx = real_size[0]; // contiguous
            const int ny = real_size[1]; // not contiguous

            std::cout << "Pos box size nx: " << nx << " ny: " << ny << std::endl;

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

            // Store meta-data in dst_plan
            dst_plan.m_position_array = position_array;
            dst_plan.m_fourier_array = fourier_array;
            /*
            int mx = 3;
            int my = 4;

            amrex::FArrayBox t_exxp_pos({{0,0,0}, {2,3,4}},1);
            amrex::BaseFab<amrex::GpuComplex<amrex::Real>> t_exp_fou({{0,0,0}, {2,3,4}},1);
            amrex::FArrayBox t_pos({{0,0,0}, {2,3,4}},1);
            amrex::FArrayBox t_fou({{0,0,0}, {2,3,4}},1);

            amrex::Array4<amrex::GpuComplex<amrex::Real>> const & test_arr = test.array();
            amrex::ParallelFor({{0,0,0}, {2,3,4}}, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    test_arr(i,j,k).m_real = i+3*j+3*4*k;
                    test_arr(i,j,k).m_imag = i+3*j+3*4*k+100;
                });
            amrex::GpuComplex<amrex::Real>* test_ptr = test.dataPtr();
            for(int i=0; i<60;++i) {
                std::cout << test_ptr[i].m_real << " ## " << test_ptr[i].m_imag << std::endl;
            }
            */

            return dst_plan;
        }
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
        cufftDestroy( dst_plan.m_plan );
        cufftDestroy( dst_plan.m_plan_b );
    }

    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("AnyDST::Execute()");

        if(!dst_plan.use_small_dst) {
            // Expand in position space m_position_array -> m_expanded_position_array
            ExpandR2R(*dst_plan.m_expanded_position_array, *dst_plan.m_position_array);
            amrex::Gpu::synchronize();

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
            cudaDeviceSynchronize();
            // Shrink in Fourier space m_expanded_fourier_array -> m_fourier_array
            ShrinkC2R(*dst_plan.m_fourier_array, *dst_plan.m_expanded_fourier_array);

            if ( result != CUFFT_SUCCESS ) {
                amrex::Print() << " forward transform using cufftExec failed ! Error: " <<
                    CuFFTUtils::cufftErrorToString(result) << "\n";
            }
            amrex::Gpu::synchronize();
        }
        else {
            const int nx = dst_plan.m_position_array->box().length(0); // initially contiguous
            const int ny = dst_plan.m_position_array->box().length(1); // contiguous after transpose

            amrex::Real* const pos_arr = dst_plan.m_position_array->dataPtr();
            amrex::GpuComplex<amrex::Real>* comp_arr = dst_plan.m_expanded_fourier_array->dataPtr();
            amrex::Real* const real_arr = dst_plan.m_expanded_position_array->dataPtr();
            amrex::Real* const fourier_arr = dst_plan.m_fourier_array->dataPtr();
            //std::cout << "######### stepp #########" << std::endl;
            //print_arr(pos_arr, nx, ny);
            ToComplex(pos_arr, comp_arr, nx, ny);
            //print_arr(comp_arr, (nx+1)/2+1, ny);
            C2Rfft(dst_plan.m_plan, comp_arr, real_arr);
            //print_arr(real_arr, nx+1, ny);
            ToSine(real_arr, pos_arr, nx, ny);
            //print_arr(pos_arr, nx, ny);
            Transpose(pos_arr, fourier_arr, nx, ny);
            //print_arr(fourier_arr, ny, nx);
            ToComplex(fourier_arr, comp_arr, ny, nx);
            //print_arr(comp_arr, (ny+1)/2+1, nx);
            C2Rfft(dst_plan.m_plan_b, comp_arr, real_arr);
            //print_arr(real_arr, ny+1, nx);
            ToSine(real_arr, pos_arr, ny, nx);
            //print_arr(pos_arr, ny, nx);
            Transpose(pos_arr, fourier_arr, ny, nx);
            //print_arr(fourier_arr, nx, ny);
        }
    }
}
