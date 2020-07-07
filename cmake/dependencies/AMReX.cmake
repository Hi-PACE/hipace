macro(find_amrex)
    if(HiPACE_amrex_internal)
        message(STATUS "Downloading AMReX ...")
        include(FetchContent)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options
        if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(ENABLE_ASSERTIONS ON CACHE BOOL "")
            # note: floating-point exceptions can slow down debug runs a lot
            set(ENABLE_FPE ON CACHE INTERNAL "")
        else()
            set(ENABLE_ASSERTIONS OFF CACHE BOOL "")
            set(ENABLE_FPE OFF CACHE INTERNAL "")
        endif()

        if(HiPACE_COMPUTE STREQUAL CUDA)
            set(ENABLE_ACC   OFF CACHE INTERNAL "")
            set(ENABLE_CUDA  ON  CACHE INTERNAL "")
            set(ENABLE_DPCPP OFF CACHE INTERNAL "")
            #set(ENABLE_HIP   OFF CACHE INTERNAL "")
            set(ENABLE_OMP   OFF CACHE INTERNAL "")
        elseif(HiPACE_COMPUTE STREQUAL OMP)
            set(ENABLE_ACC   OFF CACHE INTERNAL "")
            set(ENABLE_CUDA  OFF CACHE INTERNAL "")
            set(ENABLE_DPCPP OFF CACHE INTERNAL "")
            #set(ENABLE_HIP   OFF CACHE INTERNAL "")
            set(ENABLE_OMP   ON  CACHE INTERNAL "")
        elseif(HiPACE_COMPUTE STREQUAL DPCPP)
            set(ENABLE_ACC   OFF CACHE INTERNAL "")
            set(ENABLE_CUDA  OFF CACHE INTERNAL "")
            set(ENABLE_DPCPP ON  CACHE BOOL "")
            #set(ENABLE_HIP   OFF CACHE BOOL "")
            set(ENABLE_OMP   OFF CACHE INTERNAL "")
        else()
            set(ENABLE_ACC   OFF CACHE INTERNAL "")
            set(ENABLE_CUDA  OFF CACHE INTERNAL "")
            set(ENABLE_DPCPP OFF CACHE INTERNAL "")
            #set(ENABLE_HIP   OFF CACHE INTERNAL "")
            set(ENABLE_OMP   OFF CACHE INTERNAL "")
        endif()

        if(HiPACE_MPI)
            set(ENABLE_MPI ON CACHE INTERNAL "")
        else()
            set(ENABLE_MPI OFF CACHE INTERNAL "")
        endif()

        if(HiPACE_PRECISION STREQUAL "double")
            set(ENABLE_DP ON CACHE INTERNAL "")
            set(ENABLE_DP_PARTICLES ON CACHE INTERNAL "")
        else()
            set(ENABLE_DP OFF CACHE INTERNAL "")
            set(ENABLE_DP_PARTICLES OFF CACHE INTERNAL "")
        endif()

        set(ENABLE_FORTRAN OFF CACHE INTERNAL "")
        set(ENABLE_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
        set(ENABLE_TUTORIALS OFF CACHE INTERNAL "")
        set(ENABLE_PARTICLES ON CACHE INTERNAL "")
        set(ENABLE_TINY_PROFILE ON CACHE BOOL "")
        set(ENABLE_LINEAR_SOLVERS OFF CACHE INTERNAL "")

        # ENABLE_ASCENT
        # ENABLE_CONDUIT
        # ENABLE_SENSEI_IN_SITU
        # we'll need this for Python bindings
        #set(ENABLE_PIC ON CACHE INTERNAL "")

        set(DIM 3 CACHE INTERNAL "")

        FetchContent_Declare(fetchedamrex
            GIT_REPOSITORY ${HiPACE_amrex_repo}
            GIT_TAG        ${HiPACE_amrex_branch}
            BUILD_IN_SOURCE 0
        )
        FetchContent_GetProperties(fetchedamrex)

        if(NOT fetchedamrex_POPULATED)
            FetchContent_Populate(fetchedamrex)
            list(APPEND CMAKE_MODULE_PATH "${fetchedamrex_SOURCE_DIR}/Tools/CMake")
            if(ENABLE_CUDA)
                enable_language(CUDA)
                include(AMReX_SetupCUDA)
            endif()
            add_subdirectory(${fetchedamrex_SOURCE_DIR} ${fetchedamrex_BINARY_DIR})
        endif()

        # advanced fetch options
        mark_as_advanced(FETCHCONTENT_BASE_DIR)
        mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
        mark_as_advanced(FETCHCONTENT_QUIET)
        mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDAMREX)
        mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
        mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDAMREX)

        # AMReX options not relevant to HiPACE users
        mark_as_advanced(AMREX_BUILD_DATETIME)
        mark_as_advanced(DIM)
        mark_as_advanced(ENABLE_ACC)
        mark_as_advanced(ENABLE_ASSERTIONS)
        mark_as_advanced(ENABLE_AMRDATA)
        mark_as_advanced(ENABLE_BASE_PROFILE) # mutually exclusive to tiny profile
        mark_as_advanced(ENABLE_CONDUIT)
        mark_as_advanced(ENABLE_CUDA)
        mark_as_advanced(ENABLE_DP)
        mark_as_advanced(ENABLE_DP_PARTICLES)
        mark_as_advanced(ENABLE_DPCPP)
        mark_as_advanced(ENABLE_EB)
        mark_as_advanced(ENABLE_FPE)
        mark_as_advanced(ENABLE_FORTRAN)
        mark_as_advanced(ENABLE_FORTRAN_INTERFACES)
        mark_as_advanced(ENABLE_HDF5)  # we will do HDF5 I/O (and more) via openPMD-api
        mark_as_advanced(ENABLE_LINEAR_SOLVERS)
        mark_as_advanced(ENABLE_MEM_PROFILE)
        mark_as_advanced(ENABLE_MPI)
        mark_as_advanced(ENABLE_OMP)
        mark_as_advanced(ENABLE_PIC)
        mark_as_advanced(ENABLE_SENSEI_INSITU)
        mark_as_advanced(ENABLE_TINY_PROFILE)
        mark_as_advanced(TP_PROFILE)
        mark_as_advanced(USE_XSDK_DEFAULTS)

        message(STATUS "AMReX: Using INTERNAL version '${AMREX_PKG_VERSION}' (${AMREX_GIT_VERSION})")
    else()
        find_package(AMReX 20.05 CONFIG REQUIRED COMPONENTS 3D PARTICLES DPARTICLES DP TINYP)
        message(STATUS "AMReX: Found version '${AMReX_VERSION}'")
    endif()
endmacro()

set(HiPACE_amrex_repo "https://github.com/AMReX-Codes/amrex.git"
    CACHE STRING
    "Repository URI to pull and build AMReX from if(HiPACE_amrex_internal)")
set(HiPACE_amrex_branch "development"
    CACHE STRING
    "Repository branch for HiPACE_amrex_repo if(HiPACE_amrex_internal)")

find_amrex()
