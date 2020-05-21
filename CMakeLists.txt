# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.14.0)
project(HiPACE VERSION 0.20.2)

# list(APPEND CMAKE_MODULE_PATH "${HiPACE_SOURCE_DIR}/cmake")


# Project structure ###########################################################
#
# temporary build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CACHE PATH "Build directory for archives")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CACHE PATH "Build directory for libraries")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CACHE PATH "Build directory for binaries")

# install directories
include(GNUInstallDirs)
set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/HiPACE"
    CACHE PATH "CMake config package location for installed targets")
if(WIN32)
    set(CMAKE_INSTALL_LIBDIR Lib
        CACHE PATH "Object code libraries")
    set_property(CACHE CMAKE_INSTALL_CMAKEDIR PROPERTY VALUE "cmake")
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
option(HiPACE_USE_INTERNAL_PICSAR "Download & build PICSAR" OFF)


# Dependencies ################################################################
#
if(HiPACE_USE_INTERNAL_AMReX)
    include(FetchContent)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    #set(BUILD_TESTING OFF)
    #set(BUILD_EXAMPLES OFF)
    set(ENABLE_TUTORIALS OFF CACHE INTERNAL "")
    set(ENABLE_PARTICLES ON CACHE INTERNAL "")
    set(ENABLE_TINY_PROFILE ON CACHE INTERNAL "")
    set(AMReX_DIM 3 CACHE INTERNAL "")
    #set(AMReX_ENABLE_PIC ON CACHE INTERNAL "")
    # TEMPORARY until fixed upstream
    set(ENABLE_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
    #set(ENABLE_FORTRAN ON CACHE INTERNAL "")
    FetchContent_Declare(fetchedamrex
      GIT_REPOSITORY "https://github.com/AMReX-Codes/amrex.git"
      GIT_TAG        "development"
      BUILD_IN_SOURCE 0
    )
    #FetchContent_MakeAvailable(fetchedamrex)
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


# Targets #####################################################################
#
# executable
add_executable(HiPACE)
add_executable(HiPACE::HiPACE ALIAS HiPACE)

# own headers
target_include_directories(HiPACE PUBLIC
    $<BUILD_INTERFACE:${HiPACE_SOURCE_DIR}/Source>
#    $<BUILD_INTERFACE:${HiPACE_BINARY_DIR}/Source>
#    $<INSTALL_INTERFACE:Source>
)

# if we include <AMReX_buildInfo.H>
# include(AMReXBuildInfo)
# generate_buildinfo(HiPACE "${HiPACE_SOURCE_DIR}")

# add sources
add_subdirectory(src)

# properties
#target_compile_features(HiPACE
#    PUBLIC cxx_std_14
#)
#set_target_properties(HiPACE PROPERTIES
#    CXX_EXTENSIONS OFF
#    CXX_STANDARD_REQUIRED ON
#)

#if(ENABLE_CUDA)
#    set_cpp_sources_to_cuda_language( HiPACE )
#endif()


# FIXME hackity-hack path to PICSAR
target_include_directories(HiPACE PRIVATE
    /home/axel/src/picsar/src/multi_physics/QED/src
)

target_link_libraries(HiPACE PUBLIC AMReX::amrex)
if(ENABLE_CUDA)
    setup_target_for_cuda_compilation( HiPACE )
endif()


# Defines #####################################################################
#
target_compile_definitions(HiPACE PRIVATE -DHiPACE_DIM_3D)
target_compile_definitions(HiPACE PRIVATE -DDHiPACE_DIM_XZ)

# DEFINES += -DHiPACE_USE_PY
# DEFINES += -DHiPACE_DIM_3D
# DEFINES += -DHiPACE_DIM_RZ
# DEFINES += -DHiPACE_DIM_XZ
# CXXFLAGS += -DHiPACE_QED
# CXXFLAGS += -DHiPACE_QED_TABLE_GEN
# DEFINES += -DHiPACE_GIT_VERSION=\"$(HiPACE_GIT_VERSION)\"
# DEFINES += -DHiPACE_USE_OPENPMD
# DEFINES += -DHiPACE_USE_PSATD
# DEFINES += -DHiPACE_USE_PSATD_HYBRID
# DEFINES += -DHiPACE_DO_ELECTROSTATIC
# DEFINES += -DHiPACE_USE_HDF5


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


# Generate Files with Configuration Options ###################################
#
# TODO configure a version.hpp
#configure_file(
#    ${HiPACE_SOURCE_DIR}/include/HiPACE/config.hpp.in
#    ${HiPACE_BINARY_DIR}/include/HiPACE/config.hpp
#    @ONLY
#)

#configure_file(
#    ${HiPACE_SOURCE_DIR}/HiPACEConfig.cmake.in
#    ${HiPACE_BINARY_DIR}/HiPACEConfig.cmake
#    @ONLY
#)

# get absolute paths to linked libraries
function(HiPACEreclibs tgtname outname)
    get_target_property(PC_PRIVATE_LIBS_TGT ${tgtname} INTERFACE_LINK_LIBRARIES)
    foreach(PC_LIB IN LISTS PC_PRIVATE_LIBS_TGT)
       if(TARGET ${PC_LIB})
           HiPACEreclibs(${PC_LIB} ${outname})
       else()
           if(PC_LIB)
               string(APPEND ${outname} " ${PC_LIB}")
           endif()
       endif()
    endforeach()
    set(${outname} ${${outname}} PARENT_SCOPE)
endfunction()

if(HiPACE_HAVE_PKGCONFIG)
    HiPACEreclibs(HiPACE HiPACE_PC_PRIVATE_LIBS)
    if(BUILD_SHARED_LIBS)
        set(HiPACE_PC_STATIC false)
    else()
        set(HiPACE_PC_STATIC true)
    endif()
    configure_file(
        ${HiPACE_SOURCE_DIR}/HiPACE.pc.in
        ${HiPACE_BINARY_DIR}/HiPACE.pc
        @ONLY
    )
endif()

include(CMakePackageConfigHelpers)
write_basic_package_version_file("HiPACEConfigVersion.cmake"
    VERSION ${HiPACE_VERSION}
    COMPATIBILITY SameMajorVersion
)


# Installs ####################################################################
#
# headers, libraries and executables
set(HiPACE_INSTALL_TARGET_NAMES HiPACE)

install(TARGETS ${HiPACE_INSTALL_TARGET_NAMES}
    EXPORT HiPACETargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
#if(HiPACE_HAVE_PYTHON)
#    install(TARGETS HiPACE.py
#        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
#    )
#endif()
#install(DIRECTORY "${HiPACE_SOURCE_DIR}/include/HiPACE"
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#    FILES_MATCHING PATTERN "*.hpp"
#)
#install(
#    FILES ${HiPACE_BINARY_DIR}/include/HiPACE/config.hpp
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HiPACE
#)

# CMake package file for find_package(HiPACE::HiPACE) in depending projects
#install(EXPORT HiPACETargets
#    FILE HiPACETargets.cmake
#    NAMESPACE HiPACE::
#    DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
#)
#install(
#    FILES
#        ${HiPACE_BINARY_DIR}/HiPACEConfig.cmake
#        ${HiPACE_BINARY_DIR}/HiPACEConfigVersion.cmake
#    DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
#)
# pkg-config .pc file for depending legacy projects
#   This is for projects that do not use a build file generator, e.g.
#   because they compile manually on the command line or write their
#   Makefiles by hand.
if(HiPACE_HAVE_PKGCONFIG)
    install(
        FILES       ${HiPACE_BINARY_DIR}/HiPACE.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )
endif()


# Tests #######################################################################
#
#if(BUILD_TESTING)
#    enable_testing()
#
#    add_test(...)
#endif()


# Examples ####################################################################
#
if(BUILD_EXAMPLES)
    # ...
endif()


# Status Message for Build Options ############################################
#
message("")
message("HiPACE build configuration:")
message("  Version: ${HiPACE_VERSION}")
message("  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} "
                        "${CMAKE_CXX_COMPILER_VERSION} "
                        "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("")
message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message("        bin: ${CMAKE_INSTALL_BINDIR}")
message("        lib: ${CMAKE_INSTALL_LIBDIR}")
message("    include: ${CMAKE_INSTALL_INCLUDEDIR}")
message("      cmake: ${CMAKE_INSTALL_CMAKEDIR}")
if(HiPACE_HAVE_PYTHON)
    message("     python: ${CMAKE_INSTALL_PYTHONDIR}")
endif()
message("")
message("  Build Type: ${CMAKE_BUILD_TYPE}")
#if(BUILD_SHARED_LIBS)
#    message("  Library: shared")
#else()
#    message("  Library: static")
#endif()
message("  Testing: ${BUILD_TESTING}")
#message("  Invasive Tests: ${HiPACE_USE_INVASIVE_TESTS}")
#message("  Internal VERIFY: ${HiPACE_USE_VERIFY}")
message("  Build Options:")

foreach(opt IN LISTS HiPACE_CONFIG_OPTIONS)
  if(${HiPACE_HAVE_${opt}})
    message("    ${opt}: ON")
  else()
    message("    ${opt}: OFF")
  endif()
endforeach()
message("    DIM: ${HiPACE_DIM}")
message("")
