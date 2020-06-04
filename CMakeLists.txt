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
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    if(ENABLE_CUDA)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()


# Output Directories ##########################################################
#
# temporary build directories
set_default_build_dirs()

# install directories
set_default_install_dirs()


# Options and Variants ########################################################
#
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
    pkg_check_modules(fftw3 REQUIRED IMPORTED_TARGET fftw3)
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
if(NOT ENABLE_CUDA)
    target_link_libraries(HiPACE PUBLIC PkgConfig::fftw3)
endif()

# AMReX helper function: propagate CUDA specific target & source properties
if(ENABLE_CUDA)
    setup_target_for_cuda_compilation(HiPACE)
    target_compile_features(HiPACE PUBLIC cuda_std_14)
    set_target_properties(HiPACE PROPERTIES
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
    )
endif()


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

    if(NOT ENABLE_MPI)
        add_test(NAME Main.Examples.Serial.Inputs
                 COMMAND $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}/examples/inputs
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )
    else()
        configure_mpiexec(1)  # one rank in MPI_TEST_EXE
        add_test(NAME Main.Examples.1Rank.Inputs
                 COMMAND ${MPI_TEST_EXE} $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}/examples/inputs
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )

        configure_mpiexec(2)  # two ranks in MPI_TEST_EXE
        add_test(NAME Main.Examples.2Ranks.Inputs
                 COMMAND ${MPI_TEST_EXE} $<TARGET_FILE:HiPACE> ${HiPACE_SOURCE_DIR}/examples/inputs
                 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        )
    endif()
endif()


# Status Summary for Build Options ############################################
#
hipace_print_summary()
