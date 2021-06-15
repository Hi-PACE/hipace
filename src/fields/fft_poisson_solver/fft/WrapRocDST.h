#ifdef AMREX_USE_FLOAT
    rocfft_precision precision = rocfft_precision_single;
#else
    rocfft_precision precision = rocfft_precision_double;
#endif

    // forward declaration of error translation
    std::string rocfftErrorToString (const rocfft_status err);

    namespace {
        void assert_rocfft_status (std::string const& name, rocfft_status status)
        {
            if (status != rocfft_status_success) {
                amrex::Abort(name + " failed! Error: " + rocfftErrorToString(status));
            }
        }
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

        // Initialize fft_plan.m_plan with the vendor fft plan.
        rocfft_status result;
        result = rocfft_plan_create(&(dst_plan.m_plan), \
                                    rocfft_placement_notinplace, \
                                    rocfft_transform_type_real_forward, \
                                    precision, \
                                    dim, \
                                    lengths, \
                                    1, \
                                    nullptr);

        assert_rocfft_status("rocfft_plan_create", result);

        // Store meta-data in dst_plan
        dst_plan.m_position_array = position_array;
        dst_plan.m_fourier_array = fourier_array;

        return dst_plan;
    }

    void DestroyPlan (DSTplan& dst_plan)
    {
        rocfft_plan_destroy( dst_plan.m_plan );
    }

    void Execute (DSTplan& dst_plan){
        HIPACE_PROFILE("AnyDST::Execute()");

        // Expand in position space m_position_array -> m_expanded_position_array
        ExpandR2R(*dst_plan.m_expanded_position_array, *dst_plan.m_position_array);

        rocfft_status result;

        rocfft_execution_info execinfo = NULL;
        result = rocfft_execution_info_create(&execinfo);
        assert_rocfft_status("rocfft_execution_info_create", result);

        std::size_t buffersize = 0;
        result = rocfft_plan_get_work_buffer_size(dst_plan.m_plan, &buffersize);
        assert_rocfft_status("rocfft_plan_get_work_buffer_size", result);

        result = rocfft_execution_info_set_stream(execinfo, amrex::Gpu::gpuStream());
        assert_rocfft_status("rocfft_execution_info_set_stream", result);

        // R2C FFT m_expanded_position_array -> m_expanded_fourier_array
        // 2nd argument type still wrong, should be void*
        // reinterpret_cast<AnyFFT::Complex*>(dst_plan.m_expanded_fourier_array->dataPtr()), //3rd arg
        result = rocfft_execute(dst_plan.m_plan, \
                                (void**)&(dst_plan.m_expanded_position_array), \
                                (void**)&(dst_plan.m_expanded_fourier_array), \
                                execinfo);

        assert_rocfft_status("rocfft_execute", result);

        result = rocfft_execution_info_destroy(execinfo);
        assert_rocfft_status("rocfft_execution_info_destroy", result);

        // Shrink in Fourier space m_expanded_fourier_array -> m_fourier_array
        ShrinkC2R(*dst_plan.m_fourier_array, *dst_plan.m_expanded_fourier_array);

    }

    /** \brief This method converts a rocfftResult
     * into the corresponding string
     *
     * @param[in] err a rocfftResult
     * @return an std::string
     */
    std::string rocfftErrorToString (const rocfft_status err)
    {
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
        } else {
            return std::to_string(err) + " (unknown error code)";
        }
    }
