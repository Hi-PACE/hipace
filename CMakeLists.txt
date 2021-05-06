# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.15.0)
project(Hipace VERSION 0.20.5)

# helper functions
include(${Hipace_SOURCE_DIR}/cmake/HipaceFunctions.cmake)

# In-source tree builds are messy and can screw up the build system.
# Avoid building at least in the same dir as the root dir:
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()


# CMake policies ##############################################################
#
# CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
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
option(Hipace_MPI            "Multi-node support (message-passing)"       ON)
option(Hipace_OPENPMD        "openPMD I/O (HDF5, ADIOS)"                  ON)

set(Hipace_PRECISION_VALUES SINGLE DOUBLE)
set(Hipace_PRECISION DOUBLE CACHE STRING "Floating point precision (SINGLE/DOUBLE)")
set_property(CACHE Hipace_PRECISION PROPERTY STRINGS ${Hipace_PRECISION_VALUES})
if(NOT Hipace_PRECISION IN_LIST Hipace_PRECISION_VALUES)
    message(FATAL_ERROR "Hipace_PRECISION (${Hipace_PRECISION}) must be one of ${Hipace_PRECISION_VALUES}")
endif()

set(Hipace_COMPUTE_VALUES NOACC CUDA SYCL HIP OMP)
set(Hipace_COMPUTE NOACC CACHE STRING
    "On-node, accelerated computing backend (NOACC/CUDA/SYCL/HIP/OMP)")
set_property(CACHE Hipace_COMPUTE PROPERTY STRINGS ${Hipace_COMPUTE_VALUES})
if(NOT Hipace_COMPUTE IN_LIST Hipace_COMPUTE_VALUES)
    message(FATAL_ERROR "Hipace_COMPUTE (${Hipace_COMPUTE}) must be one of ${Hipace_COMPUTE_VALUES}")
endif()

option(Hipace_amrex_internal "Download & build AMReX" ON)

# change the default build type to RelWithDebInfo (or Release) instead of Debug
set_default_build_type("RelWithDebInfo")

# this defined the variable BUILD_TESTING which is ON by default
include(CTest)


# Dependencies ################################################################
#

# AMReX
#   builds AMReX from source (default) or finds an existing install of AMReX
include(${Hipace_SOURCE_DIR}/cmake/dependencies/AMReX.cmake)
#   suppress warnings in AMReX headers (use -isystem instead of -I)
make_third_party_includes_system(AMReX::amrex AMReX)

# FFTW (non-GPU) and cuFFT (GPU)
if(NOT Hipace_COMPUTE STREQUAL CUDA)
    find_package(PkgConfig REQUIRED QUIET)
    if(Hipace_PRECISION STREQUAL "DOUBLE")
        pkg_check_modules(fftw3 REQUIRED IMPORTED_TARGET fftw3)
    else()
        pkg_check_modules(fftw3f REQUIRED IMPORTED_TARGET fftw3f)
    endif()
endif()

# openPMD-api
include(${Hipace_SOURCE_DIR}/cmake/dependencies/openPMD.cmake)

# Targets #####################################################################
#
# executable
add_executable(Hipace)
add_executable(Hipace::Hipace ALIAS Hipace)

# own headers
target_include_directories(Hipace PRIVATE
    $<BUILD_INTERFACE:${Hipace_SOURCE_DIR}/src>
)

# if we include <AMReX_buildInfo.H> we will need to call:
#include(AMReXBuildInfo)
#generate_buildinfo(Hipace "${Hipace_SOURCE_DIR}")
#target_link_libraries(Hipace PRIVATE buildInfo::Hipace)

# add sources
add_subdirectory(src)

# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(Hipace PUBLIC cxx_std_14)
set_target_properties(Hipace PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link dependencies
target_link_libraries(Hipace PUBLIC Hipace::thirdparty::AMReX)

if(Hipace_COMPUTE STREQUAL CUDA)
    # CUDA_ADD_CUFFT_TO_TARGET(Hipace)
    target_link_libraries(Hipace PUBLIC cufft)
else()
    if(Hipace_PRECISION STREQUAL "DOUBLE")
        target_link_libraries(Hipace PUBLIC PkgConfig::fftw3)
    else()
        target_link_libraries(Hipace PUBLIC PkgConfig::fftw3f)
    endif()
endif()

# AMReX helper function: propagate CUDA specific target & source properties
if(Hipace_COMPUTE STREQUAL CUDA)
    setup_target_for_cuda_compilation(Hipace)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
        target_compile_features(Hipace PUBLIC cuda_std_14)
        set_target_properties(Hipace PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON
        )
    endif()
endif()

if(Hipace_OPENPMD)
    target_compile_definitions(Hipace PUBLIC HIPACE_USE_OPENPMD)
    target_link_libraries(Hipace PUBLIC openPMD::openPMD)
endif()

if(AMReX_LINEAR_SOLVERS)
    target_compile_definitions(Hipace PUBLIC AMREX_USE_LINEAR_SOLVERS)
endif()

# fancy binary name for build variants
set_hipace_binary_name()


# Defines #####################################################################
#
# Let's use them as sparsely as possible to avoid MxNxOxP... binary variants.
get_source_version(Hipace ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(Hipace PUBLIC HIPACE_GIT_VERSION="${Hipace_GIT_VERSION}")


# Warnings ####################################################################
#
set_cxx_warnings()


# Generate Configuration and .pc Files ########################################
#
# these files are used if Hipace is installed and picked up by a downstream
# project (not needed yet)


# Installs ####################################################################
#
# public headers, libraries and executables
install(TARGETS Hipace
    EXPORT HipaceTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


# Tests #######################################################################
#
if(BUILD_TESTING)
    enable_testing()

    if(NOT Hipace_MPI)

        add_test(NAME blowout_wake.Serial
                 COMMAND ${Hipace_SOURCE_DIR}/tests/blowout_wake.Serial.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.SI.Serial
                 COMMAND ${Hipace_SOURCE_DIR}/tests/beam_in_vacuum.SI.Serial.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.normalized.Serial
                 COMMAND ${Hipace_SOURCE_DIR}/tests/beam_in_vacuum.normalized.Serial.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

    else()

        add_test(NAME blowout_wake.2Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/blowout_wake.2Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        #add_test(NAME hosing.2Rank
        #         COMMAND ${Hipace_SOURCE_DIR}/tests/hosing.2Rank.sh
        #                 $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
        #         WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        #)

        add_test(NAME ionization.2Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/ionization.2Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME blowout_wake_explicit.2Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/blowout_wake_explicit.2Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_evolution.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/beam_evolution.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME adaptive_time_step.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/adaptive_time_step.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME grid_current.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/grid_current.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME linear_wake.normalized.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/linear_wake.normalized.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME linear_wake.SI.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/linear_wake.SI.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME gaussian_linear_wake.normalized.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/gaussian_linear_wake.normalized.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME gaussian_linear_wake.SI.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/gaussian_linear_wake.SI.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME reset.2Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/reset.2Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.SI.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/beam_in_vacuum.SI.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.normalized.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/beam_in_vacuum.normalized.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME next_deposition_beam.2Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/next_deposition_beam.2Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME slice_IO.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/slice_IO.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME gaussian_weight.1Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/gaussian_weight.1Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.normalized.2Rank
                 COMMAND ${Hipace_SOURCE_DIR}/tests/beam_in_vacuum.normalized.2Rank.sh
                         $<TARGET_FILE:Hipace> ${Hipace_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

    endif()
endif()


# Status Summary for Build Options ############################################
#
hipace_print_summary()
