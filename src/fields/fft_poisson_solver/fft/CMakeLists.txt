if (Hipace_COMPUTE STREQUAL CUDA)
  target_sources(Hipace
    PRIVATE
        WrapCuFFT.cpp
        WrapCuDST.cpp
        CuFFTUtils.cpp
  )
else()
  target_sources(Hipace
    PRIVATE
        WrapFFTW.cpp
        WrapDSTW.cpp
  )
endif()
