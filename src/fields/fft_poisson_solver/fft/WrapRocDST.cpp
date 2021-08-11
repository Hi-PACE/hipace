#include "AnyDST.H"
#include "RocFFTUtils.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_Config.H>


namespace AnyDST
{
    /** \brief Extend src into a symmetrized larger array dst
     *
     * \param[in,out] dst destination array, odd symmetry around 0 and the middle points in x and y
     * \param[in] src source array
     */
    void ExpandR2R (amrex::FArrayBox& dst, amrex::FArrayBox& src)
    {
        HIPACE_PROFILE("AnyDST::ExpandR2R()");
        /* todo */
    }

    /** \brief Extract symmetrical src array into smaller array dst
     *
     * \param[in,out] dst destination array
     * \param[in] src destination array, symmetric in x and y
     */
    void ShrinkC2R (amrex::FArrayBox& dst, amrex::BaseFab<amrex::GpuComplex<amrex::Real>>& src)
    {
        HIPACE_PROFILE("AnyDST::ShrinkC2R()");
        /* todo */
    }

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
        HIPACE_PROFILE("AnyDST::ToComplex()");
        /* todo */
    }

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
        HIPACE_PROFILE("AnyDST::C2Rfft()");
        /* todo */
    }

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
        HIPACE_PROFILE("AnyDST::ToSine()");
        /* todo */
    }

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
        HIPACE_PROFILE("AnyDST::Transpose()");
        /* todo */
    }

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array)
    {
        HIPACE_PROFILE("AnyDST::CreatePlan()");
        DSTplan dst_plan;
        const int nx = real_size[0];
        const int ny = real_size[1];
        rocfft_status status;
        int dim = 2;

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

        // Store meta-data in dst_plan
        dst_plan.m_position_array = position_array;
        dst_plan.m_fourier_array = fourier_array;

        return dst_plan;
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
        rocfft_plan_destroy( dst_plan.m_plan );
    }

    template<direction d>
    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("AnyDST::Execute()");

        // Swap position and fourier space based on execute direction
        amrex::FArrayBox* position_array =
            (d == direction::forward) ? dst_plan.m_position_array : dst_plan.m_fourier_array;
        amrex::FArrayBox* fourier_array =
            (d == direction::forward) ? dst_plan.m_fourier_array : dst_plan.m_position_array;

        // Expand in position space m_position_array -> m_expanded_position_array
        ExpandR2R(*dst_plan.m_expanded_position_array, *position_array);

        rocfft_status result;

        rocfft_execution_info execinfo = NULL;
        result = rocfft_execution_info_create(&execinfo);
        RocFFTUtils::assert_rocfft_status("rocfft_execution_info_create", result);

        std::size_t buffersize = 0;
        result = rocfft_plan_get_work_buffer_size(dst_plan.m_plan, &buffersize);
        RocFFTUtils::assert_rocfft_status("rocfft_plan_get_work_buffer_size", result);

        result = rocfft_execution_info_set_stream(execinfo, amrex::Gpu::gpuStream());
        RocFFTUtils::assert_rocfft_status("rocfft_execution_info_set_stream", result);

        // R2C FFT m_expanded_position_array -> m_expanded_fourier_array
        // 2nd argument type still wrong, should be void*
        // reinterpret_cast<AnyFFT::Complex*>(dst_plan.m_expanded_fourier_array->dataPtr()), //3rd arg
        result = rocfft_execute(dst_plan.m_plan,
                                (void**)&(dst_plan.m_expanded_position_array),
                                (void**)&(dst_plan.m_expanded_fourier_array),
                                execinfo);

        RocFFTUtils::assert_rocfft_status("rocfft_execute", result);

        result = rocfft_execution_info_destroy(execinfo);
        RocFFTUtils::assert_rocfft_status("rocfft_execution_info_destroy", result);

        // Shrink in Fourier space m_expanded_fourier_array -> m_fourier_array
        ShrinkC2R(*fourier_array, *dst_plan.m_expanded_fourier_array);

    }

    template void Execute<direction::forward>(DSTplan& dst_plan);
    template void Execute<direction::backward>(DSTplan& dst_plan);
}
