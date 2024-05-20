# Copyright 2020-2022
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
# Severin Diederichs
# License: BSD-3-Clause-LBNL

# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.18.0)
project(HiPACE VERSION 24.03)

# helper functions
include(${HiPACE_SOURCE_DIR}/cmake/HiPACEFunctions.cmake)

# In-source tree builds are messy and can screw up the build system.
# Avoid building at least in the same dir as the root dir:
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()


# CMake policies ##############################################################
#
# Setting a cmake_policy to OLD is deprecated by definition and will raise a
# verbose warning
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
endif()

# AMReX 21.06+ supports CUDA_ARCHITECTURES with CMake 3.20+
# CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

# We use simple syntax in cmake_dependent_option, so we are compatible with the
# extended syntax in CMake 3.22+
# https://cmake.org/cmake/help/v3.22/policy/CMP0127.html
if(POLICY CMP0127)
    cmake_policy(SET CMP0127 NEW)
endif()


# C++ Standard in Superbuilds #################################################
#
# This is the easiest way to push up a C++17 requirement for AMReX, PICSAR and
# openPMD-api until they increase their requirement.
set_cxx17_superbuild()


# CCache Support ##############################################################
#
# this is an optional tool that stores compiled object files; allows fast
# re-builds even with "make clean" in between. Mainly used to store AMReX
# objects
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(HiPACE_CCACHE_DEFAULT ON)
else()
    set(HiPACE_CCACHE_DEFAULT OFF)  # we are a subproject in a superbuild
endif()
option(HiPACE_CCACHE "Enable ccache for faster rebuilds" ${HiPACE_CCACHE_DEFAULT})
if(HiPACE_CCACHE)
    set_ccache()
endif()


# Output Directories ##########################################################
#
# temporary build directories
set_default_build_dirs()

# install directories
set_default_install_dirs()


# Options and Variants ########################################################
#
option(HiPACE_MPI            "Multi-node support (message-passing)"    ON)
option(HiPACE_OPENPMD        "openPMD I/O (HDF5, ADIOS)"               ON)
option(HiPACE_openpmd_mpi    "parallel version of openPMD I/O"         ${HiPACE_MPI})

set(HiPACE_PUSHER_VALUES LEAPFROG AB5)
set(HiPACE_PUSHER LEAPFROG CACHE STRING "Plasma pusher (LEAPFROG/AB5)")
set_property(CACHE HiPACE_PUSHER PROPERTY STRINGS ${HiPACE_PUSHER_VALUES})
if(NOT HiPACE_PUSHER IN_LIST HiPACE_PUSHER_VALUES)
    message(FATAL_ERROR "HiPACE_PUSHER (${HiPACE_PUSHER}) must be one of ${HiPACE_PUSHER_VALUES}")
endif()

set(HiPACE_PRECISION_VALUES SINGLE DOUBLE)
set(HiPACE_PRECISION DOUBLE CACHE STRING "Floating point precision (SINGLE/DOUBLE)")
set_property(CACHE HiPACE_PRECISION PROPERTY STRINGS ${HiPACE_PRECISION_VALUES})
if(NOT HiPACE_PRECISION IN_LIST HiPACE_PRECISION_VALUES)
    message(FATAL_ERROR "HiPACE_PRECISION (${HiPACE_PRECISION}) must be one of ${HiPACE_PRECISION_VALUES}")
endif()

set(HiPACE_COMPUTE_VALUES NOACC CUDA SYCL HIP OMP)
set(HiPACE_COMPUTE OMP CACHE STRING
    "On-node, accelerated computing backend (NOACC/CUDA/SYCL/HIP/OMP)")
set_property(CACHE HiPACE_COMPUTE PROPERTY STRINGS ${HiPACE_COMPUTE_VALUES})
if(NOT HiPACE_COMPUTE IN_LIST HiPACE_COMPUTE_VALUES)
    message(FATAL_ERROR "HiPACE_COMPUTE (${HiPACE_COMPUTE}) must be one of ${HiPACE_COMPUTE_VALUES}")
endif()

option(HiPACE_amrex_internal "Download & build AMReX" ON)

# change the default build type to Release (or RelWithDebInfo) instead of Debug
set_default_build_type("Release")

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
include(${HiPACE_SOURCE_DIR}/cmake/dependencies/FFT.cmake)

# openPMD-api
include(${HiPACE_SOURCE_DIR}/cmake/dependencies/openPMD.cmake)

# Targets #####################################################################
#
# executable
add_executable(HiPACE)
add_executable(HiPACE::HiPACE ALIAS HiPACE)

# own headers
target_include_directories(HiPACE PRIVATE
    $<BUILD_INTERFACE:${HiPACE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${HiPACE_BINARY_DIR}/src>
)

# if we include <AMReX_buildInfo.H> we will need to call:
#include(AMReXBuildInfo)
#generate_buildinfo(HiPACE "${HiPACE_SOURCE_DIR}")
#target_link_libraries(HiPACE PRIVATE buildInfo::HiPACE)

# add sources
add_subdirectory(src)

# C++ properties: at least a C++17 capable compiler is needed
target_compile_features(HiPACE PUBLIC cxx_std_17)
set_target_properties(HiPACE PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link dependencies
target_link_libraries(HiPACE PUBLIC HiPACE::thirdparty::AMReX)

target_link_libraries(HiPACE PUBLIC HiPACE::thirdparty::FFT)

# AMReX helper function: propagate CUDA specific target & source properties
if(HiPACE_COMPUTE STREQUAL CUDA)
    setup_target_for_cuda_compilation(HiPACE)
    target_compile_features(HiPACE PUBLIC cuda_std_17)
    set_target_properties(HiPACE PROPERTIES
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(HiPACE_OPENPMD)
    target_compile_definitions(HiPACE PUBLIC HIPACE_USE_OPENPMD)
    target_link_libraries(HiPACE PUBLIC openPMD::openPMD)
endif()

if(HiPACE_PUSHER STREQUAL "AB5")
    target_compile_definitions(HiPACE PUBLIC HIPACE_USE_AB5_PUSH)
endif()

if(AMReX_LINEAR_SOLVERS)
    target_compile_definitions(HiPACE PUBLIC AMREX_USE_LINEAR_SOLVERS)
endif()

# fancy binary name for build variants
set_hipace_binary_name()


# Defines #####################################################################
#
# Let's use them as sparsely as possible to avoid MxNxOxP... binary variants.


# Warnings ####################################################################
#
set_cxx_warnings()


# Generate Configuration and .pc Files ########################################
#
get_source_version(HiPACE ${HiPACE_SOURCE_DIR})
configure_file(
    ${HiPACE_SOURCE_DIR}/src/HipaceVersion.H.in
    ${HiPACE_BINARY_DIR}/src/HipaceVersion.H
    @ONLY
)

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
                 COMMAND bash ${HiPACE_SOURCE_DIR}/tests/blowout_wake.Serial.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.SI.Serial
                 COMMAND bash ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.SI.Serial.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.normalized.Serial
                 COMMAND bash ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.normalized.Serial.sh
                         $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

    else()

        # These tests run on CPU and on GPU

        add_test(NAME blowout_wake.2Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/blowout_wake.2Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_evolution.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/beam_evolution.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME adaptive_time_step.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/adaptive_time_step.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME radiation_reaction.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/radiation_reaction.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME grid_current.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/grid_current.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME linear_wake.normalized.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/linear_wake.normalized.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME linear_wake.SI.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/linear_wake.SI.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME gaussian_linear_wake.normalized.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/gaussian_linear_wake.normalized.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME gaussian_linear_wake.SI.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/gaussian_linear_wake.SI.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME reset.2Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/reset.2Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.SI.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.SI.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum.normalized.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.normalized.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME slice_IO.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/slice_IO.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME output_coarsening.2Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/output_coarsening.2Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME beam_in_vacuum_open_boundary.normalized.1Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum_open_boundary.normalized.1Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        add_test(NAME laser_evolution.SI.2Rank
                COMMAND bash ${HiPACE_SOURCE_DIR}/tests/laser_evolution.SI.2Rank.sh
                        $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        if (NOT HiPACE_COMPUTE STREQUAL CUDA)

            # These tests only run on CPU

            add_test(NAME collisions.SI.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/collisions.SI.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME collisions_beam.SI.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/collisions_beam.SI.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME ion_motion.SI.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/ion_motion.SI.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME ionization.2Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/ionization.2Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME from_file.normalized.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/from_file.normalized.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME from_file.SI.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/from_file.SI.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME restart.normalized.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/restart.normalized.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME blowout_wake_explicit.2Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/blowout_wake_explicit.2Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME laser_blowout_wake_explicit.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/laser_blowout_wake_explicit.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME laser_blowout_wake_explicit.SI.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/laser_blowout_wake_explicit.SI.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME next_deposition_beam.2Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/next_deposition_beam.2Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME gaussian_weight.1Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/gaussian_weight.1Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME beam_in_vacuum.normalized.2Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/beam_in_vacuum.normalized.2Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )

            add_test(NAME production.SI.2Rank
                    COMMAND bash ${HiPACE_SOURCE_DIR}/tests/production.SI.2Rank.sh
                            $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )
        endif()
    endif()
endif()


# Status Summary for Build Options ############################################
#
hipace_print_summary()
