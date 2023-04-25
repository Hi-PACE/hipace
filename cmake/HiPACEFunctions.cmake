# Set C++17 for the whole build if not otherwise requested
#
# This is the easiest way to push up a C++17 requirement for AMReX, PICSAR and
# openPMD-api until they increase their requirement.
#
macro(set_cxx17_superbuild)
    if(NOT DEFINED CMAKE_CXX_STANDARD)
        set(CMAKE_CXX_STANDARD 17)
    endif()
    if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
        set(CMAKE_CXX_EXTENSIONS OFF)
    endif()
    if(NOT DEFINED CMAKE_CXX_STANDARD_REQUIRED)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
    if(NOT DEFINED CMAKE_CUDA_EXTENSIONS)
        set(CMAKE_CUDA_EXTENSIONS OFF)
    endif()
    if(NOT DEFINED CMAKE_CUDA_STANDARD_REQUIRED)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endmacro()


# find the CCache tool and use it if found
#
macro(set_ccache)
    if(NOT CMAKE_CXX_COMPILER_LAUNCHER)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
            if(HiPACE_COMPUTE STREQUAL CUDA)
                set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
            endif()
            message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
        else()
            message(STATUS "Could NOT find CCache")
        endif()
        mark_as_advanced(CCACHE_PROGRAM)
    else()
        message(STATUS "Using CCache: ${CCACHE_PROGRAM}")
    endif()
endmacro()


# set names and paths of temporary build directories
# the defaults in CMake are sub-ideal for historic reasons, lets make them more
# Unix-ish and portable.
#
macro(set_default_build_dirs)
    if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                CACHE PATH "Build directory for archives")
        mark_as_advanced(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                CACHE PATH "Build directory for libraries")
        mark_as_advanced(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                CACHE PATH "Build directory for binaries")
        mark_as_advanced(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    endif()
endmacro()


# set names and paths of install directories
# the defaults in CMake are sub-ideal for historic reasons, lets make them more
# Unix-ish and portable.
#
macro(set_default_install_dirs)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
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
    endif()
endmacro()


# change the default CMAKE_BUILD_TYPE
# the default in CMake is Debug for historic reasons
#
macro(set_default_build_type default_build_type)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
        if(NOT CMAKE_BUILD_TYPE)
            set(CMAKE_BUILD_TYPE ${default_build_type}
                    CACHE STRING
                    "Choose the build type, e.g. Release, Debug, or RelWithDebInfo." FORCE)
            set_property(CACHE CMAKE_BUILD_TYPE
                    PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
        endif()
        if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
            message(WARNING "CMAKE_BUILD_TYPE '${CMAKE_BUILD_TYPE}' is not one of "
                    "${CMAKE_CONFIGURATION_TYPES}. Is this a typo?")
        endif()
    endif()
endmacro()

# Set CXX
# Note: this is a bit legacy and one should use CMake TOOLCHAINS instead.
#
macro(set_cxx_warnings)
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
endmacro()


# Take an <imported_target> and expose it as INTERFACE target with
# HiPACE::thirdparty::<propagated_name> naming and SYSTEM includes.
#
function(make_third_party_includes_system imported_target propagated_name)
    add_library(HiPACE::thirdparty::${propagated_name} INTERFACE IMPORTED)
    target_link_libraries(HiPACE::thirdparty::${propagated_name} INTERFACE ${imported_target})

    if(TARGET ${imported_target})
        get_target_property(imported_target_type ${imported_target} TYPE)
        if(NOT imported_target_type STREQUAL INTERFACE_LIBRARY)
            get_target_property(ALL_INCLUDES ${imported_target} INCLUDE_DIRECTORIES)
            if(ALL_INCLUDES)
                set_target_properties(HiPACE::thirdparty::${propagated_name} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
                target_include_directories(HiPACE::thirdparty::${propagated_name} SYSTEM INTERFACE ${ALL_INCLUDES})
            endif()
        endif()
    endif()
endfunction()



# Set a feature-based binary name for the HiPACE executable and create a generic
# hipace symlink to it. Only sets options relevant for users (see summary).
#
function(set_hipace_binary_name)
    set_target_properties(HiPACE PROPERTIES OUTPUT_NAME "hipace")

    if(HiPACE_MPI)
        set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".MPI")
    else()
        set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".NOMPI")
    endif()

    set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".${HiPACE_COMPUTE}")

    if(HiPACE_PRECISION STREQUAL "DOUBLE")
        set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".DP")
    else()
        set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".SP")
    endif()

    #if(HiPACE_ASCENT)
    #    set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".ASCENT")
    #endif()

    #if(HiPACE_OPENPMD)
    #    set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".OPMD")
    #endif()

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".DEBUG")
    endif()

    if(HiPACE_PUSHER STREQUAL "AB5")
        set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".AB5")
    else()
        set_property(TARGET HiPACE APPEND_STRING PROPERTY OUTPUT_NAME ".LF")
    endif()

    # alias to the latest build, because using the full name is often confusing
    add_custom_command(TARGET HiPACE POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:HiPACE>
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hipace
    )
endfunction()


# FUNCTION: get_source_version
#
# Retrieves source version info and sets internal cache variables
# ${NAME}_GIT_VERSION and ${NAME}_PKG_VERSION
#
function(get_source_version NAME SOURCE_DIR)
    find_package(Git QUIET)
    set(_tmp "")

    # Try to inquire software version from git
    if(EXISTS ${SOURCE_DIR}/.git AND ${GIT_FOUND})
        execute_process(COMMAND git describe --abbrev=12 --dirty --always --tags
                WORKING_DIRECTORY ${SOURCE_DIR}
                OUTPUT_VARIABLE _tmp)
        string( STRIP "${_tmp}" _tmp)
    endif()

    # Is there a CMake project version?
    # For deployed releases that build from tarballs, this is what we want to pick
    if(NOT _tmp AND ${NAME}_VERSION)
        set(_tmp "${${NAME}_VERSION}-nogit")
    endif()

    set(${NAME}_GIT_VERSION "${_tmp}" CACHE INTERNAL "")
    unset(_tmp)
endfunction ()


# Prints a summary of HiPACE options at the end of the CMake configuration
#
function(hipace_print_summary)
    message("")
    message("HiPACE build configuration:")
    message("  Version: ${HiPACE_VERSION} (${HiPACE_GIT_VERSION})")
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
    set(BLD_TYPE_UNKNOWN "")
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND
       NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
        set(BLD_TYPE_UNKNOWN " (unknown type, check warning)")
    endif()
    message("  Build type: ${CMAKE_BUILD_TYPE}${BLD_TYPE_UNKNOWN}")
    #if(BUILD_SHARED_LIBS)
    #    message("  Library: shared")
    #else()
    #    message("  Library: static")
    #endif()
    message("  Testing: ${BUILD_TESTING}")
    message("  Build options:")
    message("    COMPUTE: ${HiPACE_COMPUTE}")
    message("    MPI: ${HiPACE_MPI}")
    message("    OPENPMD: ${HiPACE_OPENPMD}")
    message("    PRECISION: ${HiPACE_PRECISION}")
    message("    PUSHER: ${HiPACE_PUSHER}")
    message("")
endfunction()
