macro(find_amrex)
    if(HiPACE_amrex_src)
        message(STATUS "Compiling local AMReX ...")
        message(STATUS "AMReX source path: ${HiPACE_amrex_src}")
        if(NOT IS_DIRECTORY ${HiPACE_amrex_src})
            message(FATAL_ERROR "Specified directory HiPACE_amrex_src='${HiPACE_amrex_src}' does not exist!")
        endif()
    elseif(HiPACE_amrex_internal)
        message(STATUS "Downloading AMReX ...")
        message(STATUS "AMReX repository: ${HiPACE_amrex_repo} (${HiPACE_amrex_branch})")
        include(FetchContent)
    endif()
    if(HiPACE_amrex_internal OR HiPACE_amrex_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options
        if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(AMReX_ASSERTIONS ON CACHE BOOL "")
            # note: floating-point exceptions can slow down debug runs a lot
            set(AMReX_FPE ON CACHE INTERNAL "")
        else()
            set(AMReX_ASSERTIONS OFF CACHE BOOL "")
            set(AMReX_FPE OFF CACHE INTERNAL "")
        endif()

        if(HiPACE_COMPUTE STREQUAL OMP)
            set(AMReX_GPU_BACKEND  "NONE" CACHE INTERNAL "")
            set(AMReX_OMP          ON     CACHE INTERNAL "")
        elseif(HiPACE_COMPUTE STREQUAL NOACC)
            set(AMReX_GPU_BACKEND  "NONE" CACHE INTERNAL "")
            set(AMReX_OMP          OFF    CACHE INTERNAL "")
        else()
            set(AMReX_GPU_BACKEND  "${HiPACE_COMPUTE}" CACHE INTERNAL "")
            set(AMReX_OMP          OFF                 CACHE INTERNAL "")
        endif()

        if(HiPACE_MPI)
            set(AMReX_MPI ON CACHE INTERNAL "")
        else()
            set(AMReX_MPI OFF CACHE INTERNAL "")
        endif()

        if(HiPACE_PRECISION STREQUAL "DOUBLE")
            set(AMReX_PRECISION "DOUBLE" CACHE INTERNAL "")
            set(AMReX_PARTICLES_PRECISION "DOUBLE" CACHE INTERNAL "")
        else()
            set(AMReX_PRECISION "SINGLE" CACHE INTERNAL "")
            set(AMReX_PARTICLES_PRECISION "SINGLE" CACHE INTERNAL "")
        endif()

        set(AMReX_AMRLEVEL OFF CACHE INTERNAL "")
        set(AMReX_ENABLE_TESTS OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
        set(AMReX_BUILD_TUTORIALS OFF CACHE INTERNAL "")
        set(AMReX_PARTICLES ON CACHE INTERNAL "")
        set(AMReX_TINY_PROFILE ON CACHE BOOL "")
        set(AMReX_LINEAR_SOLVERS OFF CACHE INTERNAL "")

        # we don't need RDC and disabling it simplifies the build
        # complexity and potentially improves code optimization
        # note: once we add & enable ASCENT and SENSEI, we need RDC
        set(AMReX_GPU_RDC OFF CACHE BOOL "")

        # AMReX_ASCENT
        # AMReX_CONDUIT
        # AMReX_SENSEI
        # we'll need this for Python bindings
        #set(AMReX_PIC ON CACHE INTERNAL "")

        set(AMReX_SPACEDIM 3 CACHE INTERNAL "")

        if(HiPACE_amrex_src)
            list(APPEND CMAKE_MODULE_PATH "${HiPACE_amrex_src}/Tools/CMake")
            if(HiPACE_COMPUTE STREQUAL CUDA)
                enable_language(CUDA)
            endif()
            add_subdirectory(${HiPACE_amrex_src} _deps/localamrex-build/)
        else()
            if(HiPACE_COMPUTE STREQUAL CUDA)
                enable_language(CUDA)
            endif()
            FetchContent_Declare(fetchedamrex
                GIT_REPOSITORY ${HiPACE_amrex_repo}
                GIT_TAG        ${HiPACE_amrex_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedamrex)
            list(APPEND CMAKE_MODULE_PATH "${fetchedamrex_SOURCE_DIR}/Tools/CMake")

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDAMREX)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDAMREX)
        endif()

        # AMReX options not relevant to HiPACE users
        mark_as_advanced(AMREX_BUILD_DATETIME)
        mark_as_advanced(AMReX_CONDUIT)
        mark_as_advanced(AMReX_ENABLE_TESTS)
        mark_as_advanced(AMReX_SPACEDIM)
        mark_as_advanced(AMReX_ASSERTIONS)
        mark_as_advanced(AMReX_AMRDATA)
        mark_as_advanced(AMReX_BASE_PROFILE) # mutually exclusive to tiny profile
        mark_as_advanced(AMReX_DPCPP)
        mark_as_advanced(AMReX_CUDA)
        mark_as_advanced(AMReX_GPU_RDC)
        mark_as_advanced(AMReX_PARTICLES_PRECISION)
        mark_as_advanced(AMReX_PRECISION)
        mark_as_advanced(AMReX_EB)
        mark_as_advanced(AMReX_FPE)
        mark_as_advanced(AMReX_FORTRAN)
        mark_as_advanced(AMReX_FORTRAN_INTERFACES)
        mark_as_advanced(AMReX_HDF5)  # we do HDF5 I/O (and more) via openPMD-api
        mark_as_advanced(AMReX_LINEAR_SOLVERS)
        mark_as_advanced(AMReX_MEM_PROFILE)
        mark_as_advanced(AMReX_MPI)
        mark_as_advanced(AMReX_MPI_THREAD_MULTIPLE)
        mark_as_advanced(AMReX_OMP)
        mark_as_advanced(AMReX_PIC)
        mark_as_advanced(AMReX_SENSEI)
        mark_as_advanced(AMReX_TINY_PROFILE)
        mark_as_advanced(AMReX_TP_PROFILE)
        mark_as_advanced(USE_XSDK_DEFAULTS)

        message(STATUS "AMReX: Using version '${AMREX_PKG_VERSION}' (${AMREX_GIT_VERSION})") 
    else()
        message(STATUS "Searching for pre-installed AMReX ...")
        set(COMPONENT_PRECISION ${HiPACE_PRECISION} P${HiPACE_PRECISION})
        find_package(AMReX 24.08 CONFIG REQUIRED COMPONENTS 3D ${COMPONENT_PRECISION} PARTICLES)
        # note: TINYP skipped because user-configured and optional

        # AMReX CMake helper scripts
        list(APPEND CMAKE_MODULE_PATH "${AMReX_DIR}/AMReXCMakeModules")

        message(STATUS "AMReX: Found version '${AMReX_VERSION}'")
    endif()
endmacro()

# local source-tree
set(HiPACE_amrex_src ""
    CACHE PATH
    "Local path to AMReX source directory (preferred if set)")

# Git fetcher
set(HiPACE_amrex_repo "https://github.com/AMReX-Codes/amrex.git"
    CACHE STRING
    "Repository URI to pull and build AMReX from if(HiPACE_amrex_internal)")
set(HiPACE_amrex_branch "development"
    CACHE STRING
    "Repository branch for HiPACE_amrex_repo if(HiPACE_amrex_internal)")

find_amrex()
