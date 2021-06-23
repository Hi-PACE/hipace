#!/bin/bash
#/5fau3pj is openmpi
#/6z7nqa is hip/rocm
#/imi7cyn is llvm-amdgpu

source /software/spack/share/spack/setup-env.sh
spack load gcc && spack load /5fau3pj && spack load rocfft && spack load rocrand && spack load rocprim && spack load rocm-clang-ocl && spack load /6z7nqa4 && spack load openpmd-api
#&& spack load /imi7cyn

mkdir -p build && cd build
cmake3 .. -DHiPACE_COMPUTE=HIP -DAMReX_AMD_ARCH=gfx900 -DCMAKE_CXX_COMPILER=$(which hipcc) -DCMAKE_C_COMPILER=$(which clang) -DHiPACE_OPENPMD=OFF
cmake3 --build . -j 4
