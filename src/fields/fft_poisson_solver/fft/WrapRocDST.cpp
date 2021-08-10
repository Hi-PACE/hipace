#include "AnyDST.H"
#include "RocFFTUtils.H"
#include "utils/HipaceProfilerWrapper.H"

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
        /* todo */
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
        /* todo */
    }

    template<direction d>
    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("AnyDST::Execute()");

        /* todo */
    }

    template void Execute<direction::forward>(DSTplan& dst_plan);
    template void Execute<direction::backward>(DSTplan& dst_plan);
}
