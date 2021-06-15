if (HiPACE_COMPUTE STREQUAL CUDA)
  target_sources(HiPACE
    PRIVATE
        WrapCuFFT.cpp
        CuFFTUtils.cpp
  )
elseif (HiPACE_COMPUTE STREQUAL HIP)
  target_sources(HiPACE
      PRIVATE
        WrapRocFFT.cpp
  )
else()
  target_sources(HiPACE
    PRIVATE
        WrapFFTW.cpp
  )
endif()
target_sources(HiPACE
    PRIVATE
        WrapGPUDST.cpp
    )
