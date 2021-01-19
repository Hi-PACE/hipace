if (HiPACE_COMPUTE STREQUAL CUDA)
  target_sources(HiPACE
    PRIVATE
        WrapCuFFT.cpp
        WrapCuDST.cpp
        CuFFTUtils.cpp
  )
else()
  target_sources(HiPACE
    PRIVATE
        WrapFFTW.cpp
        WrapDSTW.cpp
  )
endif()
