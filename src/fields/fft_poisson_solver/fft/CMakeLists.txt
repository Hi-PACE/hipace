if (ENABLE_CUDA)
  target_sources(HiPACE
    PRIVATE
        WrapCuFFT.cpp
  )
else()
  target_sources(HiPACE
    PRIVATE
        WrapFFTW.cpp
  )
endif()
