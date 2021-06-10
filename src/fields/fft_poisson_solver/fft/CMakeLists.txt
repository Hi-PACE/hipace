if (HiPACE_COMPUTE STREQUAL CUDA)
  target_sources(HiPACE
    PRIVATE
        WrapCuFFT.cpp
        WrapCuDST.cpp
        CuFFTUtils.cpp
  )
elseif (HiPACE_COMPUTE STREQUAL HIP)
  target_sources(HiPACE
      PRIVATE
        WrapRocFFT.cpp
        WrapRocDST.cpp
  )
else()
  target_sources(HiPACE
    PRIVATE
        WrapFFTW.cpp
        WrapDSTW.cpp
  )
endif()
