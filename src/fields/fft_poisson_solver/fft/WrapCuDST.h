#ifdef AMREX_USE_FLOAT
    cufftType VendorR2C = CUFFT_R2C;
#else
    cufftType VendorR2C = CUFFT_D2Z;
#endif

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array)
    {
        HIPACE_PROFILE("AnyDST::CreatePlan()");
        DSTplan dst_plan;
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

    void DestroyPlan (DSTplan& dst_plan)
    {
        cufftDestroy( dst_plan.m_plan );
    }

    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("AnyDST::Execute()");

        // Expand in position space m_position_array -> m_expanded_position_array
        ExpandR2R(*dst_plan.m_expanded_position_array, *dst_plan.m_position_array);

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
        ShrinkC2R(*dst_plan.m_fourier_array, *dst_plan.m_expanded_fourier_array);

        if ( result != CUFFT_SUCCESS ) {
            amrex::Print() << " forward transform using cufftExec failed ! Error: " <<
                CuFFTUtils::cufftErrorToString(result) << "\n";
        }
    }
