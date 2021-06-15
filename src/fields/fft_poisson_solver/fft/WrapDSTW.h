#ifdef AMREX_USE_FLOAT
    const auto VendorCreatePlanR2R2D = fftwf_plan_r2r_2d;
#else
    const auto VendorCreatePlanR2R2D = fftw_plan_r2r_2d;
#endif

    DSTplan CreatePlan (const amrex::IntVect& real_size, amrex::FArrayBox* position_array,
                        amrex::FArrayBox* fourier_array)
    {
        DSTplan dst_plan;
        const int nx = real_size[0];
        const int ny = real_size[1];

        // Initialize fft_plan.m_plan with the vendor fft plan.
        // Swap dimensions: AMReX FAB are Fortran-order but FFTW is C-order
        dst_plan.m_plan = VendorCreatePlanR2R2D(
            ny, nx, position_array->dataPtr(), fourier_array->dataPtr(),
            FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);

        // Store meta-data in fft_plan
        dst_plan.m_position_array = position_array;
        dst_plan.m_fourier_array = fourier_array;

        amrex::Box expanded_position_box {{0, 0, 0}, {2*nx+1, 2*ny+1, 0}};
        amrex::Box expanded_fourier_box {{0, 0, 0}, {nx+1, 2*ny+1, 0}};
        dst_plan.m_expanded_position_array =std::make_unique<
            amrex::FArrayBox>(expanded_position_box, 1);
        dst_plan.m_expanded_fourier_array = std::make_unique<
            amrex::BaseFab<amrex::GpuComplex<amrex::Real>>>(expanded_fourier_box, 1);

        return dst_plan;
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
#  ifdef AMREX_USE_FLOAT
        fftwf_destroy_plan( dst_plan.m_plan );
#  else
        fftw_destroy_plan( dst_plan.m_plan );
#  endif
    }

    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("AnyDST::Execute()");
#  ifdef AMREX_USE_FLOAT
        fftwf_execute( dst_plan.m_plan );
#  else
        fftw_execute( dst_plan.m_plan );
#  endif
    }
