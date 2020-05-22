# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.14.0)
project(HiPACE VERSION 0.20.5)

include(${HiPACE_SOURCE_DIR}/cmake/HiPACEFunctions.cmake)
# list(INSERT CMAKE_MODULE_PATH 0 "${HiPACE_SOURCE_DIR}/cmake")


# Output Directories ##########################################################
#
# temporary build directories
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        CACHE PATH "Build directory for archives")
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        CACHE PATH "Build directory for libraries")
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        CACHE PATH "Build directory for binaries")
endif()

# install directories
include(GNUInstallDirs)
if(NOT CMAKE_INSTALL_CMAKEDIR)
    set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/HiPACE"
        CACHE PATH "CMake config package location for installed targets")
    if(WIN32)
        set(CMAKE_INSTALL_LIBDIR Lib
            CACHE PATH "Object code libraries")
        set_property(CACHE CMAKE_INSTALL_CMAKEDIR PROPERTY VALUE "cmake")
    endif()
    mark_as_advanced(CMAKE_INSTALL_CMAKEDIR)
endif()


# Options and Variants ########################################################
#

# change the default build type to Release instead of Debug
set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the build type, e.g. Release or Debug." FORCE)
endif()

function(HiPACE_option name description default)
    set(HiPACE_USE_${name} ${default} CACHE STRING "${description}")
    set_property(CACHE HiPACE_USE_${name} PROPERTY
        STRINGS "ON;TRUE;AUTO;OFF;FALSE"
    )
    if(HiPACE_HAVE_${name})
        set(HiPACE_HAVE_${name} TRUE)
    else()
        set(HiPACE_HAVE_${name})
    endif()
    # list of all possible options
    set(HiPACE_CONFIG_OPTIONS ${HiPACE_CONFIG_OPTIONS} ${name} PARENT_SCOPE)
endfunction()

option(HiPACE_USE_INTERNAL_AMReX  "Download & build AMReX"  ON)
# option(HiPACE_USE_INTERNAL_PICSAR "Download & build PICSAR" OFF)


# Dependencies ################################################################
#
if(HiPACE_USE_INTERNAL_AMReX)
    include(FetchContent)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    #set(BUILD_TESTING OFF)
    #set(BUILD_EXAMPLES OFF)
    set(ENABLE_FORTRAN OFF CACHE INTERNAL "")
    set(ENABLE_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
    set(ENABLE_TUTORIALS OFF CACHE INTERNAL "")
    set(ENABLE_PARTICLES ON CACHE INTERNAL "")
    set(ENABLE_TINY_PROFILE ON CACHE INTERNAL "")
    set(AMReX_DIM 3 CACHE INTERNAL "")
    # we'll need this for Python bindings
    #set(ENABLE_PIC ON CACHE INTERNAL "")

    FetchContent_Declare(fetchedamrex
      GIT_REPOSITORY "https://github.com/AMReX-Codes/amrex.git"
      GIT_TAG        "development"
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

    message(STATUS "AMReX: Using INTERNAL version '${AMREX_PKG_VERSION}' (${AMREX_GIT_VERSION})")
else()
    find_package(AMReX 20.05 CONFIG REQUIRED COMPONENTS 3D PARTICLES DPARTICLES DP TINYP)
    message(STATUS "AMReX: Found version '${AMReX_VERSION}'")
endif()

# suppress warnings in AMReX headers
make_third_party_includes_system(AMReX::amrex AMReX)


# Targets #####################################################################
#
# executable
add_executable(HiPACE)
add_executable(HiPACE::HiPACE ALIAS HiPACE)

# own headers
target_include_directories(HiPACE PUBLIC
    $<BUILD_INTERFACE:${HiPACE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${HiPACE_BINARY_DIR}/src>
    $<INSTALL_INTERFACE:src>
)

# if we include <AMReX_buildInfo.H>
# include(AMReXBuildInfo)
# generate_buildinfo(HiPACE "${HiPACE_SOURCE_DIR}")

# add sources
add_subdirectory(src)

# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(HiPACE
    PUBLIC cxx_std_14
)
set_target_properties(HiPACE PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link dependencies
target_link_libraries(HiPACE PUBLIC HiPACE::thirdparty::AMReX)

# AMReX helper function: propagate CUDA specific target & source properties
if(ENABLE_CUDA)
    setup_target_for_cuda_compilation(HiPACE)
endif()


# Defines #####################################################################
#
# Let's use them as sparsely as possible to avoid MxNxOxP... binary variants.
#target_compile_definitions(HiPACE PRIVATE -DHiPACE_DIM_3D)


# Warnings ####################################################################
#
# TODO: LEGACY! Use CMake TOOLCHAINS instead!
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # list(APPEND CMAKE_CXX_FLAGS "-fsanitize=address") # address, memory, undefined
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    # set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
    # set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address")

    # note: might still need a
    #   export LD_PRELOAD=libclang_rt.asan.so
    # or on Debian 9 with Clang 6.0
    #   export LD_PRELOAD=/usr/lib/llvm-6.0/lib/clang/6.0.0/lib/linux/libclang_rt.asan-x86_64.so:
    #                     /usr/lib/llvm-6.0/lib/clang/6.0.0/lib/linux/libclang_rt.ubsan_minimal-x86_64.so
    # at runtime when used with symbol-hidden code (e.g. pybind11 module)

    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wextra-semi -Wunreachable-code")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wunreachable-code")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Warning C4503: "decorated name length exceeded, name was truncated"
    # Symbols longer than 4096 chars are truncated (and hashed instead)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd4503")
    # Yes, you should build against the same C++ runtime and with same
    # configuration (Debug/Release). MSVC does inconvenient choices for their
    # developers, so be it. (Our Windows-users use conda-forge builds, which
    # are consistent.)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd4251")
endif ()


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
# this defined the variable BUILD_TESTING which is ON by default
include(CTest)

if(BUILD_TESTING)
    enable_testing()

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


# Status Summary for Build Options ############################################
#
hipace_print_summary()
