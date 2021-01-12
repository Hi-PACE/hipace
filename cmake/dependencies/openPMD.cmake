function(find_openpmd)
    if(HiPACE_openpmd_internal)
        message(STATUS "Downloading openPMD-api ...")
        include(FetchContent)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://openpmd-api.readthedocs.io/en/0.13.1-alpha/dev/buildoptions.html
        set(openPMD_USE_MPI         ${HiPACE_MPI} CACHE INTERNAL "")
        set(openPMD_USE_PYTHON      OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_CLI_TOOLS OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_EXAMPLES  OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_TESTING   OFF           CACHE INTERNAL "")
        set(openPMD_INSTALL ${BUILD_SHARED_LIBS}  CACHE INTERNAL "")

        FetchContent_Declare(fetchedopenpmd
            GIT_REPOSITORY ${HiPACE_openpmd_repo}
            GIT_TAG        ${HiPACE_openpmd_branch}
            BUILD_IN_SOURCE 0
        )
        FetchContent_GetProperties(fetchedopenpmd)

        if(NOT fetchedopenpmd_POPULATED)
            FetchContent_Populate(fetchedopenpmd)
            add_subdirectory(${fetchedopenpmd_SOURCE_DIR} ${fetchedopenpmd_BINARY_DIR})
        endif()

        # advanced fetch options
        mark_as_advanced(FETCHCONTENT_BASE_DIR)
        mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
        mark_as_advanced(FETCHCONTENT_QUIET)
        mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDOPENPMD)
        mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
        mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDOPENPMD)

        # openPMD options not relevant to HiPACE users
        mark_as_advanced(openPMD_USE_INTERNAL_VARIANT)
        mark_as_advanced(openPMD_USE_INTERNAL_CATCH)
        mark_as_advanced(openPMD_USE_INTERNAL_PYBIND11)
        mark_as_advanced(openPMD_USE_INTERNAL_JSON)
        mark_as_advanced(openPMD_HAVE_PKGCONFIG)
        mark_as_advanced(openPMD_USE_INVASIVE_TESTS)
        mark_as_advanced(openPMD_USE_VERIFY)
        mark_as_advanced(ADIOS2_DOR)
        mark_as_advanced(ADIOS_CONFIG)
        mark_as_advanced(HDF5_DIR)
        mark_as_advanced(JSON_MultipleHeaders)

        message(STATUS "openPMD-api: Using INTERNAL version '${HiPACE_openpmd_branch}'")
    else()
        if(HiPACE_MPI)
            set(COMPONENT_WMPI MPI)
        else()
            set(COMPONENT_WMPI NOMPI)
        endif()
        find_package(openPMD 0.13.0 CONFIG REQUIRED COMPONENTS ${COMPONENT_WMPI})
        message(STATUS "openPMD-api: Found version '${openPMD_VERSION}'")
    endif()
endfunction()

if(HiPACE_OPENPMD)
    option(HiPACE_openpmd_internal   "Download & build openPMD-api" ON)
    set(HiPACE_openpmd_repo "https://github.com/openPMD/openPMD-api.git"
        CACHE STRING
        "Repository URI to pull and build openPMD-api from if(HiPACE_openpmd_internal)")
    set(HiPACE_openpmd_branch "a9022ee30fe640a5ca1d92c30d0658cf2bfebec6"
        CACHE STRING
        "Repository branch for HiPACE_openpmd_repo if(HiPACE_openpmd_internal)")

    set(HiPACE_HAVE_OPENPMD TRUE)
    find_openpmd()
endif()
