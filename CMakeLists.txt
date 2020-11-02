# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.14.0)
project(HiPACE VERSION 0.20.5)

# helper functions
include(${HiPACE_SOURCE_DIR}/cmake/HiPACEFunctions.cmake)

# In-source tree builds are messy and can screw up the build system.
# Avoid building at least in the same dir as the root dir:
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()


# CCache Support ##############################################################
#
# this is an optional tool that stores compiled object files; allows fast
# re-builds even with "make clean" in between. Mainly used to store AMReX
# objects
set_ccache()


# Output Directories ##########################################################
#
# temporary build directories
set_default_build_dirs()

# install directories
set_default_install_dirs()


# Options and Variants ########################################################
#
option(HiPACE_MPI            "Multi-node support (message-passing)"       ON)

set(HiPACE_PRECISION_VALUES double single)
set(HiPACE_PRECISION double CACHE STRING "Floating point precision (single/double)")
set_property(CACHE HiPACE_PRECISION PROPERTY STRINGS ${HiPACE_PRECISION_VALUES})
if(NOT HiPACE_PRECISION IN_LIST HiPACE_PRECISION_VALUES)
    message(FATAL_ERROR "HiPACE_PRECISION (${HiPACE_PRECISION}) must be one of ${HiPACE_PRECISION_VALUES}")
endif()

set(HiPACE_COMPUTE_VALUES NOACC CUDA OMP DPCPP) # HIP
set(HiPACE_COMPUTE NOACC CACHE STRING
    "On-node, accelerated computing backend (NOACC/OMP/CUDA/DPCPP)")
set_property(CACHE HiPACE_COMPUTE PROPERTY STRINGS ${HiPACE_COMPUTE_VALUES})
if(NOT HiPACE_COMPUTE IN_LIST HiPACE_COMPUTE_VALUES)
    message(FATAL_ERROR "HiPACE_COMPUTE (${HiPACE_COMPUTE}) must be one of ${HiPACE_COMPUTE_VALUES}")
endif()

option(HiPACE_amrex_internal "Download & build AMReX" ON)

# change the default build type to RelWithDebInfo (or Release) instead of Debug
set_default_build_type("RelWithDebInfo")

# this defined the variable BUILD_TESTING which is ON by default
include(CTest)


# Dependencies ################################################################
#

# AMReX
#   builds AMReX from source (default) or finds an existing install of AMReX
include(${HiPACE_SOURCE_DIR}/cmake/dependencies/AMReX.cmake)
#   suppress warnings in AMReX headers (use -isystem instead of -I)
make_third_party_includes_system(AMReX::amrex AMReX)

# FFTW (non-GPU) and cuFFT (GPU)
if(NOT ENABLE_CUDA)
    find_package(PkgConfig REQUIRED QUIET)
    if(HiPACE_PRECISION STREQUAL "double")
        pkg_check_modules(fftw3 REQUIRED IMPORTED_TARGET fftw3)
    else()
        pkg_check_modules(fftw3f REQUIRED IMPORTED_TARGET fftw3f)
    endif()
endif()

# openPMD-api


# Targets #####################################################################
#
# executable
add_executable(HiPACE)
add_executable(HiPACE::HiPACE ALIAS HiPACE)

# own headers
target_include_directories(HiPACE PRIVATE
    $<BUILD_INTERFACE:${HiPACE_SOURCE_DIR}/src>
)

# if we include <AMReX_buildInfo.H> we will need to call:
#include(AMReXBuildInfo)
#generate_buildinfo(HiPACE "${HiPACE_SOURCE_DIR}")

# add sources
add_subdirectory(src)

# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(HiPACE PUBLIC cxx_std_14)
set_target_properties(HiPACE PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link dependencies
target_link_libraries(HiPACE PUBLIC HiPACE::thirdparty::AMReX)

if(ENABLE_CUDA)
    # CUDA_ADD_CUFFT_TO_TARGET(HiPACE)
    target_link_libraries(HiPACE PUBLIC cufft)
else()
    if(HiPACE_PRECISION STREQUAL "double")
        target_link_libraries(HiPACE PUBLIC PkgConfig::fftw3)
    else()
        target_link_libraries(HiPACE PUBLIC PkgConfig::fftw3f)
    endif()
endif()

# AMReX helper function: propagate CUDA specific target & source properties
if(ENABLE_CUDA)
    setup_target_for_cuda_compilation(HiPACE)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
        target_compile_features(HiPACE PUBLIC cuda_std_14)
        set_target_properties(HiPACE PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON
        )
    endif()
endif()

# fancy binary name for build variants
set_hipace_binary_name()


# Defines #####################################################################
#
# Let's use them as sparsely as possible to avoid MxNxOxP... binary variants.
#target_compile_definitions(HiPACE PRIVATE -DHiPACE_FFTW)


# Warnings ####################################################################
#
set_cxx_warnings()


# Generate Configuration and .pc Files ########################################
#
# these files are used if HiPACE is installed and picked up by a downstream
# project (not needed yet)


# Installs ####################################################################
#
# public headers, libraries and executables
install(TARGETS HiPACE
    EXPORT HiPACETargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


# Tests #######################################################################
#
if(BUILD_TESTING)
    enable_testing()

    if(NOT HiPACE_MPI)

        add_test(NAME blowout_wake.Serial
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/blowout_wake.Serial.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.SI.Serial
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.SI.Serial.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.normalized.Serial
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.normalized.Serial.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

    else()

        configure_mpiexec(1)  # one rank in MPI_TEST_EXE
        add_test(NAME blowout_wake.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/blowout_wake.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_evolution.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/beam_evolution.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME linear_wake.normalized.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/linear_wake.normalized.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME linear_wake.SI.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/linear_wake.SI.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME gaussian_linear_wake.normalized.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/gaussian_linear_wake.normalized.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME gaussian_linear_wake.SI.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/gaussian_linear_wake.SI.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME slice_deposition.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/slice_deposition.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME reset.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/reset.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.SI.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.SI.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.normalized.1Rank
                 COMMAND ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.normalized.1Rank.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        # This test is temporarily removed, due to bugs in the parallelization
        # configure_mpiexec(2)  # two ranks in MPI_TEST_EXE
        # add_test(NAME blowout_wake.2Rank
        #         COMMAND ${HiPACE_SOURCE_DIR}/tests/blowout_wake.2Rank.sh
        #                 $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
        #         WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        #)

    endif()
endif()


# Status Summary for Build Options ############################################
#
hipace_print_summary()
